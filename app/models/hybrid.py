"""Typed, explainable hybrid GraphRAG retrieval models."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.models.graph import GraphEntity, GraphProvenance, RelationshipEvidence
from app.models.vector import VectorRetrievalResult


class RetrievalSource(StrEnum):
    """Signals that caused a chunk to enter the retrieval context."""

    VECTOR = "VECTOR"
    GRAPH_ENTITY = "GRAPH_ENTITY"
    GRAPH_RELATIONSHIP = "GRAPH_RELATIONSHIP"
    RELATIONSHIP_EVIDENCE = "RELATIONSHIP_EVIDENCE"


class GraphExpandedEntity(BaseModel):
    """An entity reached from one or more vector seed chunks."""

    entity: GraphEntity
    graph_distance: int = Field(ge=0, le=2)
    seed_chunk_ids: list[str] = Field(default_factory=list)


class GraphExpandedRelationship(BaseModel):
    """A semantic edge reached during bounded graph expansion."""

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    description: str | None = None
    graph_distance: int = Field(ge=1, le=2)
    evidence: list[RelationshipEvidence] = Field(default_factory=list)


class GraphSupportingChunk(BaseModel):
    """A source chunk found through entity mentions or edge evidence."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int = Field(ge=0)
    source: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    graph_distance: int = Field(ge=0, le=2)
    entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    retrieval_sources: list[RetrievalSource] = Field(default_factory=list)

    @property
    def provenance(self) -> GraphProvenance:
        """Expose citation-ready source provenance."""
        return GraphProvenance(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            source=self.source,
            chunk_index=self.chunk_index,
            source_metadata=self.source_metadata,
        )


class GraphExpansionResult(BaseModel):
    """Bounded graph context related to vector seed chunks."""

    entities: list[GraphExpandedEntity] = Field(default_factory=list)
    relationships: list[GraphExpandedRelationship] = Field(default_factory=list)
    supporting_chunks: list[GraphSupportingChunk] = Field(default_factory=list)


class RetrievalScores(BaseModel):
    """Independent and fused signals for one final context chunk."""

    vector_score: float | None = Field(default=None, ge=0.0, le=1.0)
    graph_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)


class HybridContextChunk(BaseModel):
    """One deduplicated, ranked chunk with retrieval explanations."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int = Field(ge=0)
    source: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_sources: list[RetrievalSource]
    scores: RetrievalScores
    graph_distance: int | None = Field(default=None, ge=0, le=2)
    related_entity_ids: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)


class HybridRetrievalRequest(BaseModel):
    """Retrieval-only request with safe caller-controlled bounds."""

    query: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    graph_hops: int = Field(default=1, ge=1, le=2)


class HybridRetrievalResult(BaseModel):
    """Structured GraphRAG context ready for Phase 6 consumption."""

    query: str
    vector_seed_results: list[VectorRetrievalResult]
    entities: list[GraphExpandedEntity]
    relationships: list[GraphExpandedRelationship]
    context: list[HybridContextChunk]
    graph_evidence_found: bool
