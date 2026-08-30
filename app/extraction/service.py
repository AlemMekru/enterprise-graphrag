"""Normalize and validate provider graph candidates."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from pydantic import ValidationError

from app.extraction.base import GraphExtractionProvider
from app.extraction.exceptions import GraphValidationError, MalformedExtractionError
from app.extraction.normalization import (
    deterministic_entity_id,
    deterministic_relationship_id,
    normalize_display_name,
    normalize_entity_name,
    normalize_entity_type,
    normalize_relationship_type,
)
from app.models.document import DocumentChunk
from app.models.graph import (
    Entity,
    ExtractionResult,
    GraphProvenance,
    ProviderExtraction,
    Relationship,
)

logger = logging.getLogger(__name__)


class GraphExtractionService:
    """Extract validated graph candidates from Phase 1 chunks."""

    def __init__(self, provider: GraphExtractionProvider) -> None:
        self.provider = provider

    def extract_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        """Extract, normalize, deduplicate, and validate one chunk."""
        raw_response = self.provider.extract(chunk.text)
        try:
            extraction = ProviderExtraction.model_validate(raw_response)
        except ValidationError as exc:
            raise MalformedExtractionError(
                "Provider response does not match the extraction schema"
            ) from exc

        provenance = self._provenance(chunk)
        entities, entity_by_name = self._normalize_entities(extraction, provenance)
        relationships = self._normalize_relationships(
            extraction, entity_by_name, provenance
        )
        result = ExtractionResult(
            entities=entities,
            relationships=relationships,
            source=provenance,
        )
        logger.info(
            "Extracted graph candidates",
            extra={
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
        )
        return result

    def extract_chunks(
        self, chunks: Iterable[DocumentChunk]
    ) -> list[ExtractionResult]:
        """Extract each chunk independently without corpus-wide resolution."""
        return [self.extract_chunk(chunk) for chunk in chunks]

    @staticmethod
    def _normalize_entities(
        extraction: ProviderExtraction,
        provenance: GraphProvenance,
    ) -> tuple[list[Entity], dict[str, Entity]]:
        entities: list[Entity] = []
        by_name: dict[str, Entity] = {}
        for candidate in extraction.entities:
            display_name = normalize_display_name(candidate.name)
            normalized_name = normalize_entity_name(display_name)
            entity_type = normalize_entity_type(candidate.entity_type)
            existing = by_name.get(normalized_name)
            if existing is not None:
                if existing.entity_type != entity_type:
                    raise GraphValidationError(
                        f"Entity {display_name!r} has conflicting types "
                        f"{existing.entity_type.value} and {entity_type.value}"
                    )
                continue

            entity = Entity(
                entity_id=deterministic_entity_id(entity_type, normalized_name),
                name=display_name,
                normalized_name=normalized_name,
                entity_type=entity_type,
                description=_normalize_optional_text(candidate.description),
                provenance=provenance,
            )
            by_name[normalized_name] = entity
            entities.append(entity)
        return entities, by_name

    @staticmethod
    def _normalize_relationships(
        extraction: ProviderExtraction,
        entity_by_name: dict[str, Entity],
        provenance: GraphProvenance,
    ) -> list[Relationship]:
        relationships: list[Relationship] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in extraction.relationships:
            source_name = normalize_entity_name(candidate.source_entity)
            target_name = normalize_entity_name(candidate.target_entity)
            source = entity_by_name.get(source_name)
            target = entity_by_name.get(target_name)
            if source is None or target is None:
                missing = candidate.source_entity if source is None else candidate.target_entity
                raise GraphValidationError(
                    f"Relationship references missing entity: {missing!r}"
                )
            if source.entity_id == target.entity_id:
                raise GraphValidationError(
                    f"Self-referential relationship is not allowed for {source.name!r}"
                )

            relationship_type = normalize_relationship_type(
                candidate.relationship_type
            )
            key = (source.entity_id, relationship_type, target.entity_id)
            if key in seen:
                continue
            seen.add(key)
            relationships.append(
                Relationship(
                    relationship_id=deterministic_relationship_id(
                        source.entity_id,
                        relationship_type,
                        target.entity_id,
                        provenance.chunk_id,
                    ),
                    source_entity=source.entity_id,
                    target_entity=target.entity_id,
                    relationship_type=relationship_type,
                    description=_normalize_optional_text(candidate.description),
                    evidence=_normalize_optional_text(candidate.evidence),
                    provenance=provenance,
                )
            )
        return relationships

    @staticmethod
    def _provenance(chunk: DocumentChunk) -> GraphProvenance:
        source = (
            chunk.source_metadata.get("source_path")
            or chunk.source_metadata.get("source")
            or chunk.source_metadata.get("filename")
        )
        return GraphProvenance(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            source=str(source) if source is not None else None,
            chunk_index=chunk.chunk_index,
            source_metadata=chunk.source_metadata,
        )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None
