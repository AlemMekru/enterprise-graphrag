"""Tests for graph extraction validation and orchestration."""

import pytest

from app.extraction.exceptions import (
    ExtractionProviderError,
    GraphValidationError,
    MalformedExtractionError,
)
from app.extraction.service import GraphExtractionService
from app.models.document import DocumentChunk
from app.models.graph import ProviderEntity, ProviderExtraction, ProviderRelationship


class StaticExtractionProvider:
    def __init__(self, response: object) -> None:
        self.response = response
        self.inputs: list[str] = []

    def extract(self, text: str) -> ProviderExtraction:
        self.inputs.append(text)
        return self.response  # type: ignore[return-value]


class FailingExtractionProvider:
    def extract(self, text: str) -> ProviderExtraction:
        raise ExtractionProviderError("provider unavailable")


def _chunk(chunk_id: str = "chunk_1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        text="The Security Team manages the Identity Access System.",
        chunk_index=2,
        source_metadata={
            "filename": "policy.md",
            "source_path": "/data/policy.md",
            "department": "Security",
        },
    )


def _valid_extraction() -> ProviderExtraction:
    return ProviderExtraction(
        entities=[
            ProviderEntity(
                name="Security Team",
                entity_type="BUSINESS_UNIT",
                description="The security function.",
            ),
            ProviderEntity(
                name="Identity Access System",
                entity_type="SYSTEM",
            ),
        ],
        relationships=[
            ProviderRelationship(
                source_entity="Security Team",
                target_entity="Identity Access System",
                relationship_type="manages",
                evidence="Security Team manages the Identity Access System.",
            )
        ],
    )


def test_service_normalizes_entities_and_relationships() -> None:
    result = GraphExtractionService(
        StaticExtractionProvider(_valid_extraction())
    ).extract_chunk(_chunk())

    assert [entity.normalized_name for entity in result.entities] == [
        "security team",
        "identity access system",
    ]
    assert result.relationships[0].relationship_type == "MANAGES"
    assert result.relationships[0].source_entity == result.entities[0].entity_id
    assert result.relationships[0].target_entity == result.entities[1].entity_id


def test_service_preserves_complete_provenance() -> None:
    result = GraphExtractionService(
        StaticExtractionProvider(_valid_extraction())
    ).extract_chunk(_chunk())

    assert result.source.document_id == "doc_1"
    assert result.source.chunk_id == "chunk_1"
    assert result.source.chunk_index == 2
    assert result.source.source == "/data/policy.md"
    assert result.source.source_metadata["department"] == "Security"
    assert result.entities[0].provenance == result.source
    assert result.relationships[0].provenance == result.source


def test_duplicate_entities_and_relationships_are_removed() -> None:
    extraction = _valid_extraction()
    extraction.entities.append(
        ProviderEntity(name="  SECURITY   TEAM ", entity_type="business unit")
    )
    extraction.relationships.append(
        ProviderRelationship(
            source_entity="security team",
            target_entity="Identity Access System",
            relationship_type="MANAGES",
            evidence="Duplicate evidence",
        )
    )

    result = GraphExtractionService(
        StaticExtractionProvider(extraction)
    ).extract_chunk(_chunk())

    assert len(result.entities) == 2
    assert len(result.relationships) == 1


def test_entity_and_relationship_ids_repeat_for_same_input() -> None:
    service = GraphExtractionService(StaticExtractionProvider(_valid_extraction()))

    first = service.extract_chunk(_chunk())
    second = service.extract_chunk(_chunk())

    assert [entity.entity_id for entity in first.entities] == [
        entity.entity_id for entity in second.entities
    ]
    assert first.relationships[0].relationship_id == (
        second.relationships[0].relationship_id
    )


def test_empty_extraction_is_valid() -> None:
    result = GraphExtractionService(
        StaticExtractionProvider(ProviderExtraction())
    ).extract_chunk(_chunk())

    assert result.entities == []
    assert result.relationships == []


def test_malformed_provider_response_is_rejected() -> None:
    service = GraphExtractionService(
        StaticExtractionProvider({"entities": "not-a-list"})
    )

    with pytest.raises(MalformedExtractionError):
        service.extract_chunk(_chunk())


def test_unknown_entity_type_is_rejected() -> None:
    extraction = ProviderExtraction(
        entities=[ProviderEntity(name="Security Team", entity_type="TEAM")]
    )

    with pytest.raises(GraphValidationError, match="Unknown entity type"):
        GraphExtractionService(
            StaticExtractionProvider(extraction)
        ).extract_chunk(_chunk())


def test_relationship_with_missing_entity_is_rejected() -> None:
    extraction = _valid_extraction()
    extraction.relationships[0].target_entity = "Unknown System"

    with pytest.raises(GraphValidationError, match="missing entity"):
        GraphExtractionService(
            StaticExtractionProvider(extraction)
        ).extract_chunk(_chunk())


def test_self_referential_relationship_is_rejected() -> None:
    extraction = _valid_extraction()
    extraction.relationships[0].target_entity = "Security Team"

    with pytest.raises(GraphValidationError, match="Self-referential"):
        GraphExtractionService(
            StaticExtractionProvider(extraction)
        ).extract_chunk(_chunk())


def test_conflicting_duplicate_entity_types_are_rejected() -> None:
    extraction = _valid_extraction()
    extraction.entities.append(
        ProviderEntity(name="security team", entity_type="ORGANIZATION")
    )

    with pytest.raises(GraphValidationError, match="conflicting types"):
        GraphExtractionService(
            StaticExtractionProvider(extraction)
        ).extract_chunk(_chunk())


def test_provider_failure_is_preserved() -> None:
    with pytest.raises(ExtractionProviderError, match="unavailable"):
        GraphExtractionService(FailingExtractionProvider()).extract_chunk(_chunk())


def test_service_extracts_multiple_chunks_independently() -> None:
    provider = StaticExtractionProvider(_valid_extraction())

    results = GraphExtractionService(provider).extract_chunks(
        [_chunk("chunk_1"), _chunk("chunk_2")]
    )

    assert len(results) == 2
    assert [result.source.chunk_id for result in results] == ["chunk_1", "chunk_2"]
    assert len(provider.inputs) == 2
