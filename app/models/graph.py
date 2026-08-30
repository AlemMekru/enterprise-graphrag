"""Typed graph extraction models and enterprise taxonomy."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    """Controlled, extensible taxonomy for enterprise entities."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    BUSINESS_UNIT = "BUSINESS_UNIT"
    LOCATION = "LOCATION"
    SYSTEM = "SYSTEM"
    APPLICATION = "APPLICATION"
    POLICY = "POLICY"
    REGULATION = "REGULATION"
    PROCESS = "PROCESS"
    PRODUCT = "PRODUCT"
    SERVICE = "SERVICE"
    EVENT = "EVENT"
    DATE = "DATE"
    CONCEPT = "CONCEPT"
    OTHER = "OTHER"


class GraphProvenance(BaseModel):
    """Traceability from a graph candidate to its source chunk."""

    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source: str | None = None
    chunk_index: int = Field(ge=0)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """Validated entity candidate ready for future graph persistence."""

    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    normalized_name: str = Field(min_length=1)
    entity_type: EntityType
    description: str | None = None
    provenance: GraphProvenance


class Relationship(BaseModel):
    """Validated relationship candidate between extracted entities."""

    relationship_id: str = Field(min_length=1)
    source_entity: str = Field(min_length=1)
    target_entity: str = Field(min_length=1)
    relationship_type: str = Field(min_length=1)
    description: str | None = None
    evidence: str | None = None
    provenance: GraphProvenance


class ExtractionResult(BaseModel):
    """Normalized graph candidates extracted from one source chunk."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    source: GraphProvenance


class ProviderEntity(BaseModel):
    """Schema-constrained entity returned by an extraction provider."""

    name: str
    entity_type: str
    description: str | None = None


class ProviderRelationship(BaseModel):
    """Schema-constrained relationship returned by an extraction provider."""

    source_entity: str
    target_entity: str
    relationship_type: str
    description: str | None = None
    evidence: str | None = None


class ProviderExtraction(BaseModel):
    """Provider response schema before deterministic normalization."""

    entities: list[ProviderEntity] = Field(default_factory=list)
    relationships: list[ProviderRelationship] = Field(default_factory=list)


class GraphExtractionRequest(BaseModel):
    """Extraction-only API request with optional caller provenance."""

    text: str = Field(min_length=1, max_length=100_000)
    document_id: str | None = Field(default=None, min_length=1)
    chunk_id: str | None = Field(default=None, min_length=1)
    chunk_index: int = Field(default=0, ge=0)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
