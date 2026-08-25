"""Document models shared by ingestion and downstream GraphRAG stages."""

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Normalized source document with stable identity and provenance."""

    document_id: str
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Ordered document segment ready for indexing or graph extraction."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int = Field(ge=0)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
