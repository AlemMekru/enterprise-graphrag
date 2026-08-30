"""Tests for Phase 3 graph candidate models."""

import pytest
from pydantic import ValidationError

from app.models.graph import Entity, EntityType, GraphProvenance, Relationship


def _provenance() -> GraphProvenance:
    return GraphProvenance(
        document_id="doc_1",
        chunk_id="chunk_1",
        source="policy.md",
        chunk_index=0,
    )


def test_entity_model_accepts_controlled_type() -> None:
    entity = Entity(
        entity_id="entity_1",
        name="Security Operations",
        normalized_name="security operations",
        entity_type=EntityType.BUSINESS_UNIT,
        provenance=_provenance(),
    )

    assert entity.entity_type is EntityType.BUSINESS_UNIT


def test_entity_model_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        Entity(
            entity_id="entity_1",
            name="Security Operations",
            normalized_name="security operations",
            entity_type="TEAM",
            provenance=_provenance(),
        )


def test_relationship_model_requires_nonempty_endpoints() -> None:
    with pytest.raises(ValidationError):
        Relationship(
            relationship_id="relationship_1",
            source_entity="",
            target_entity="entity_2",
            relationship_type="MANAGES",
            provenance=_provenance(),
        )
