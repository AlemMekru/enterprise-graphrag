"""Load supported files into normalized document records."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.ingestion.exceptions import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from app.models.document import Document


class DocumentLoader:
    """Load UTF-8 text, Markdown, and PDF documents from disk."""

    supported_extensions = frozenset({".txt", ".md", ".pdf"})

    def load(self, source: str | Path) -> Document:
        """Read one file and return a normalized, traceable document."""
        path = Path(source).expanduser()
        suffix = path.suffix.lower()

        if suffix not in self.supported_extensions:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{suffix or '<none>'}' for {path}"
            )
        if not path.is_file():
            raise DocumentReadError(f"Document does not exist or is not a file: {path}")

        resolved_path = path.resolve()
        if suffix == ".pdf":
            content, type_metadata = self._read_pdf(resolved_path)
        else:
            content, type_metadata = self._read_text(resolved_path), {}

        content = self._normalize_text(content)
        if not content:
            raise EmptyDocumentError(f"Document contains no extractable text: {path}")

        metadata: dict[str, Any] = {
            "filename": resolved_path.name,
            "source_path": str(resolved_path),
            "file_type": suffix.removeprefix("."),
            "size_bytes": resolved_path.stat().st_size,
            **type_metadata,
        }
        document_id = self._document_id(str(resolved_path), content)
        return Document(
            document_id=document_id,
            source=str(resolved_path),
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise DocumentReadError(f"Unable to read text document: {path}") from exc

    @staticmethod
    def _read_pdf(path: Path) -> tuple[str, dict[str, Any]]:
        try:
            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
        except (OSError, PyPdfError, ValueError) as exc:
            raise DocumentReadError(f"Unable to parse PDF document: {path}") from exc

        return "\n\n".join(pages), {"page_count": len(reader.pages)}

    @staticmethod
    def _normalize_text(content: str) -> str:
        return content.replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _document_id(source: str, content: str) -> str:
        digest = hashlib.sha256(f"{source}\0{content}".encode("utf-8")).hexdigest()
        return f"doc_{digest}"
