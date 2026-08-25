"""Deterministic document chunking."""

from __future__ import annotations

import hashlib

from app.ingestion.exceptions import InvalidChunkConfigurationError
from app.models.document import Document, DocumentChunk


class TextChunker:
    """Split documents into deterministic, overlapping character windows."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise InvalidChunkConfigurationError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise InvalidChunkConfigurationError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigurationError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Return ordered chunks while preserving document provenance."""
        step = self.chunk_size - self.chunk_overlap
        chunks: list[DocumentChunk] = []

        for chunk_index, start in enumerate(range(0, len(document.content), step)):
            end = min(start + self.chunk_size, len(document.content))
            text = document.content[start:end]
            chunk_id = self._chunk_id(document.document_id, chunk_index, start, text)
            source_metadata = {
                **document.metadata,
                "source": document.source,
                "start_char": start,
                "end_char": end,
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=text,
                    chunk_index=chunk_index,
                    source_metadata=source_metadata,
                )
            )
            if end == len(document.content):
                break

        return chunks

    @staticmethod
    def _chunk_id(document_id: str, index: int, start: int, text: str) -> str:
        payload = f"{document_id}\0{index}\0{start}\0{text}".encode("utf-8")
        return f"chunk_{hashlib.sha256(payload).hexdigest()}"
