"""Embedding and vector retrieval domain models."""

from typing import Any

from pydantic import BaseModel, Field

from app.models.document import DocumentChunk


class EmbeddedChunk(DocumentChunk):
    """Phase 1 chunk enriched with its embedding vector."""

    embedding: list[float] = Field(min_length=1)


class VectorRetrievalRequest(BaseModel):
    """Semantic retrieval request."""

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)


class VectorRetrievalResult(BaseModel):
    """One scored chunk returned from Neo4j vector search."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class VectorRetrievalResponse(BaseModel):
    """Structured semantic retrieval response."""

    query: str
    results: list[VectorRetrievalResult]
