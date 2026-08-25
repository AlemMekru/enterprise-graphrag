"""Composable document ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import DocumentLoader
from app.models.document import Document, DocumentChunk


class DocumentIngestionPipeline:
    """Load a source document and produce downstream-ready chunks."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        if chunker is None:
            runtime_settings = settings or get_settings()
            chunker = TextChunker(
                chunk_size=runtime_settings.chunk_size,
                chunk_overlap=runtime_settings.chunk_overlap,
            )
        self.chunker = chunker

    def ingest(self, source: str | Path) -> tuple[Document, list[DocumentChunk]]:
        """Load and chunk one source without embedding or graph side effects."""
        document = self.loader.load(source)
        return document, self.chunker.chunk(document)
