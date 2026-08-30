"""Tests for deterministic graph normalization and IDs."""

import pytest

from app.extraction.exceptions import GraphValidationError
from app.extraction.normalization import (
    deterministic_entity_id,
    deterministic_relationship_id,
    normalize_display_name,
    normalize_entity_name,
    normalize_entity_type,
    normalize_relationship_type,
)
from app.models.graph import EntityType


def test_entity_name_normalization_is_deterministic() -> None:
    assert normalize_display_name("  Microsoft   Corporation ") == (
        "Microsoft Corporation"
    )
    assert normalize_entity_name("  MICROSOFT   Corporation ") == (
        "microsoft corporation"
    )


def test_entity_type_normalization_uses_controlled_taxonomy() -> None:
    assert normalize_entity_type(" business unit ") is EntityType.BUSINESS_UNIT


def test_unknown_entity_type_is_rejected() -> None:
    with pytest.raises(GraphValidationError, match="Unknown entity type"):
        normalize_entity_type("TEAM")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("reports to", "REPORTS_TO"),
        ("governed-by", "GOVERNED_BY"),
        ("CUSTOM relationship", "CUSTOM_RELATIONSHIP"),
    ],
)
def test_relationship_type_normalization_is_extensible(
    raw: str, expected: str
) -> None:
    assert normalize_relationship_type(raw) == expected


def test_entity_ids_are_stable() -> None:
    first = deterministic_entity_id(EntityType.SYSTEM, "identity access system")
    second = deterministic_entity_id(EntityType.SYSTEM, "identity access system")

    assert first == second
    assert first.startswith("entity_")


def test_relationship_ids_are_stable_per_source_chunk() -> None:
    first = deterministic_relationship_id(
        "entity_source", "MANAGES", "entity_target", "chunk_1"
    )
    second = deterministic_relationship_id(
        "entity_source", "MANAGES", "entity_target", "chunk_1"
    )

    assert first == second
    assert first.startswith("relationship_")
