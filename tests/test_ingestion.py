"""Unit tests for document loading and deterministic chunking."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.ingestion.chunker import TextChunker
from app.ingestion.exceptions import (
    DocumentReadError,
    EmptyDocumentError,
    InvalidChunkConfigurationError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.loader import DocumentLoader
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.models.document import Document


def _write_minimal_pdf(path: Path, text: str) -> None:
    """Write a small standards-compliant PDF fixture without extra test packages."""
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(pdf)


def _document(content: str = "abcdefghij") -> Document:
    return Document(
        document_id="doc_test",
        source="/documents/policy.txt",
        content=content,
        metadata={"filename": "policy.txt", "department": "Security"},
    )


def test_load_txt_document(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("Access is reviewed quarterly.\r\n", encoding="utf-8")

    document = DocumentLoader().load(path)

    assert document.content == "Access is reviewed quarterly."
    assert document.source == str(path.resolve())
    assert document.document_id.startswith("doc_")
    assert document.metadata["filename"] == "policy.txt"
    assert document.metadata["file_type"] == "txt"
    assert document.metadata["source_path"] == str(path.resolve())


def test_load_markdown_document(tmp_path: Path) -> None:
    path = tmp_path / "handbook.md"
    path.write_text("# Handbook\n\nEmployees report incidents promptly.", encoding="utf-8")

    document = DocumentLoader().load(path)

    assert document.content.startswith("# Handbook")
    assert "report incidents" in document.content
    assert document.metadata["file_type"] == "md"


def test_load_pdf_document(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    _write_minimal_pdf(path, "Quarterly risk report")

    document = DocumentLoader().load(path)

    assert "Quarterly risk report" in document.content
    assert document.metadata["file_type"] == "pdf"
    assert document.metadata["page_count"] == 1


def test_chunking_is_deterministic() -> None:
    chunker = TextChunker(chunk_size=4, chunk_overlap=1)

    first = chunker.chunk(_document())
    second = chunker.chunk(_document())

    assert first == second
    assert [chunk.chunk_index for chunk in first] == [0, 1, 2]
    assert [chunk.text for chunk in first] == ["abcd", "defg", "ghij"]
    assert all(chunk.chunk_id.startswith("chunk_") for chunk in first)


def test_chunk_overlap_is_exact() -> None:
    chunks = TextChunker(chunk_size=5, chunk_overlap=2).chunk(_document("abcdefghijkl"))

    assert [chunk.text for chunk in chunks] == ["abcde", "defgh", "ghijk", "jkl"]
    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert chunks[1].text[-2:] == chunks[2].text[:2]


def test_chunks_preserve_document_identity_and_source_metadata() -> None:
    document = _document("enterprise controls")

    chunks = TextChunker(chunk_size=10, chunk_overlap=2).chunk(document)

    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.source_metadata["source"] == document.source for chunk in chunks)
    assert all(chunk.source_metadata["filename"] == "policy.txt" for chunk in chunks)
    assert all(chunk.source_metadata["department"] == "Security" for chunk in chunks)
    assert chunks[0].source_metadata["start_char"] == 0
    assert chunks[0].source_metadata["end_char"] == 10


def test_pipeline_loads_and_chunks_document(tmp_path: Path) -> None:
    path = tmp_path / "controls.txt"
    path.write_text("abcdefgh", encoding="utf-8")
    pipeline = DocumentIngestionPipeline(chunker=TextChunker(5, 2))

    document, chunks = pipeline.ingest(path)

    assert document.content == "abcdefgh"
    assert [chunk.text for chunk in chunks] == ["abcde", "defgh"]


def test_chunk_configuration_is_loaded_by_settings() -> None:
    settings = Settings(chunk_size=512, chunk_overlap=64, _env_file=None)

    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_settings_reject_invalid_chunk_configuration(
    chunk_size: int, chunk_overlap: int
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            _env_file=None,
        )


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_chunk_configuration(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(InvalidChunkConfigurationError):
        TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_unsupported_file_type(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_text("id,value", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        DocumentLoader().load(path)


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_empty_text_document(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"empty{suffix}"
    path.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        DocumentLoader().load(path)


def test_pdf_without_extractable_text_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    _write_minimal_pdf(path, "   ")

    with pytest.raises(EmptyDocumentError):
        DocumentLoader().load(path)


def test_malformed_text_document(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(DocumentReadError):
        DocumentLoader().load(path)


def test_malformed_pdf_document(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a PDF")

    with pytest.raises(DocumentReadError):
        DocumentLoader().load(path)


def test_missing_document() -> None:
    with pytest.raises(DocumentReadError):
        DocumentLoader().load("missing.txt")
