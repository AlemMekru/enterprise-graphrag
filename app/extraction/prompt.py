"""Centralized prompt for structured enterprise graph extraction."""

from app.models.graph import EntityType

ENTITY_TYPE_VALUES = ", ".join(entity_type.value for entity_type in EntityType)

GRAPH_EXTRACTION_SYSTEM_PROMPT = f"""
You extract grounded graph candidates from enterprise document chunks.

Rules:
- Extract only entities explicitly supported by the supplied text.
- Extract only relationships explicitly supported by the supplied text.
- Never invent facts or infer unsupported relationships.
- Use concise canonical display names and classify them consistently.
- Entity types must be one of: {ENTITY_TYPE_VALUES}.
- Relationship endpoints must exactly match names in the entities list.
- Use concise UPPER_SNAKE_CASE relationship types. Prefer common labels such as
  OWNS, MANAGES, OPERATES, USES, DEPENDS_ON, PART_OF, REPORTS_TO, GOVERNED_BY,
  APPLIES_TO, REQUIRES, PROVIDES, LOCATED_IN, or RELATED_TO when accurate; use a
  new precise UPPER_SNAKE_CASE label when none fits.
- Include short verbatim or closely faithful evidence when the text supports it.
- Do not emit self-referential relationships.
- Return empty entity and relationship lists when no meaningful graph facts exist.
""".strip()
