"""Shared Neo4j schema names used by vector and knowledge-graph stores."""

DOCUMENT_CONSTRAINT = "document_id_unique"
CHUNK_CONSTRAINT = "chunk_id_unique"
ENTITY_CONSTRAINT = "entity_id_unique"

DOCUMENT_CONSTRAINT_CYPHER = (
    f"CREATE CONSTRAINT {DOCUMENT_CONSTRAINT} IF NOT EXISTS "
    "FOR (document:Document) REQUIRE document.document_id IS UNIQUE"
)
CHUNK_CONSTRAINT_CYPHER = (
    f"CREATE CONSTRAINT {CHUNK_CONSTRAINT} IF NOT EXISTS "
    "FOR (chunk:Chunk) REQUIRE chunk.chunk_id IS UNIQUE"
)
ENTITY_CONSTRAINT_CYPHER = (
    f"CREATE CONSTRAINT {ENTITY_CONSTRAINT} IF NOT EXISTS "
    "FOR (entity:Entity) REQUIRE entity.entity_id IS UNIQUE"
)
