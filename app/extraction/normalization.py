"""Deterministic normalization and identity helpers."""

import hashlib
import re
import unicodedata

from app.extraction.exceptions import GraphValidationError
from app.models.graph import EntityType

KNOWN_RELATIONSHIP_TYPES = frozenset(
    {
        "OWNS",
        "MANAGES",
        "OPERATES",
        "USES",
        "DEPENDS_ON",
        "PART_OF",
        "REPORTS_TO",
        "GOVERNED_BY",
        "APPLIES_TO",
        "REQUIRES",
        "PROVIDES",
        "LOCATED_IN",
        "RELATED_TO",
    }
)


def normalize_display_name(value: str) -> str:
    """Trim and collapse whitespace while preserving display casing."""
    normalized = " ".join(value.split())
    if not normalized:
        raise GraphValidationError("Entity names must not be empty")
    return normalized


def normalize_entity_name(value: str) -> str:
    """Create a Unicode-normalized, case-insensitive identity key."""
    display_name = normalize_display_name(value)
    return unicodedata.normalize("NFKC", display_name).casefold()


def normalize_entity_type(value: str) -> EntityType:
    """Convert a provider type to the controlled enterprise taxonomy."""
    normalized = normalize_relationship_type(value)
    try:
        return EntityType(normalized)
    except ValueError as exc:
        raise GraphValidationError(f"Unknown entity type: {value!r}") from exc


def normalize_relationship_type(value: str) -> str:
    """Normalize relationship labels to extensible UPPER_SNAKE_CASE."""
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    normalized = re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_")
    if not normalized or not re.fullmatch(r"[A-Z][A-Z0-9_]*", normalized):
        raise GraphValidationError(f"Invalid relationship type: {value!r}")
    return normalized


def deterministic_entity_id(entity_type: EntityType, normalized_name: str) -> str:
    """Create a stable entity ID from type and normalized identity."""
    payload = f"{entity_type.value}\0{normalized_name}".encode("utf-8")
    return f"entity_{hashlib.sha256(payload).hexdigest()}"


def deterministic_relationship_id(
    source_entity_id: str,
    relationship_type: str,
    target_entity_id: str,
    chunk_id: str,
) -> str:
    """Create a stable ID for one source-supported relationship claim."""
    payload = (
        f"{source_entity_id}\0{relationship_type}\0{target_entity_id}\0{chunk_id}"
    ).encode("utf-8")
    return f"relationship_{hashlib.sha256(payload).hexdigest()}"
