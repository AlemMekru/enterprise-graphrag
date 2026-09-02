"""Transactional Neo4j knowledge-graph persistence and querying."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any

from neo4j import Driver, Transaction
from neo4j.exceptions import Neo4jError

from app.extraction.exceptions import GraphValidationError
from app.extraction.normalization import (
    deterministic_entity_id,
    deterministic_relationship_id,
    normalize_entity_name,
    normalize_relationship_type,
)
from app.graph.exceptions import (
    EntityNotFoundError,
    KnowledgeGraphInputError,
    KnowledgeGraphStoreError,
    MissingChunkError,
    MissingEntityEndpointError,
)
from app.graph.schema import (
    CHUNK_CONSTRAINT_CYPHER,
    DOCUMENT_CONSTRAINT_CYPHER,
    ENTITY_CONSTRAINT_CYPHER,
)
from app.models.graph import (
    EntityNeighborhood,
    ExtractionResult,
    GraphConnection,
    GraphEntity,
    GraphPersistenceResponse,
    RelationshipEvidence,
)
from app.models.hybrid import (
    GraphExpandedEntity,
    GraphExpandedRelationship,
    GraphExpansionResult,
    GraphSupportingChunk,
    RetrievalSource,
)

logger = logging.getLogger(__name__)

ENTITY_UPSERT_CYPHER = """
UNWIND $entities AS entity_data
MERGE (entity:Entity {entity_id: entity_data.entity_id})
ON CREATE SET entity.name = entity_data.name,
              entity.normalized_name = entity_data.normalized_name,
              entity.entity_type = entity_data.entity_type,
              entity.description = entity_data.description
SET entity.description = CASE
    WHEN entity.description IS NULL OR trim(entity.description) = ''
    THEN entity_data.description
    ELSE entity.description
END
"""

MENTION_UPSERT_CYPHER = """
MATCH (chunk:Chunk {chunk_id: $chunk_id})
UNWIND $entities AS entity_data
MATCH (entity:Entity {entity_id: entity_data.entity_id})
MERGE (chunk)-[mention:MENTIONS]->(entity)
SET mention.document_id = $document_id,
    mention.chunk_id = $chunk_id,
    mention.chunk_index = $chunk_index,
    mention.source = $source,
    mention.source_metadata_json = $source_metadata_json
"""

CHUNK_LOOKUP_CYPHER = """
MATCH (document:Document {document_id: $document_id})
      -[:HAS_CHUNK]->(chunk:Chunk {chunk_id: $chunk_id})
RETURN chunk.chunk_id AS chunk_id
"""


class Neo4jKnowledgeGraphStore:
    """Persist validated extractions and query entity neighborhoods."""

    def __init__(self, driver: Driver, database: str) -> None:
        self.driver = driver
        self.database = database

    def ensure_schema(self) -> None:
        """Idempotently create shared graph identity constraints."""
        try:
            with self.driver.session(database=self.database) as session:
                session.run(DOCUMENT_CONSTRAINT_CYPHER).consume()
                session.run(CHUNK_CONSTRAINT_CYPHER).consume()
                session.run(ENTITY_CONSTRAINT_CYPHER).consume()
        except Neo4jError as exc:
            raise KnowledgeGraphStoreError(
                "Unable to initialize Neo4j knowledge-graph schema"
            ) from exc

    def persist_extraction_result(
        self, result: ExtractionResult
    ) -> GraphPersistenceResponse:
        """Persist one complete extraction in a single write transaction."""
        self._validate_result(result)
        payload = self._persistence_payload(result)
        try:
            with self.driver.session(database=self.database) as session:
                session.execute_write(self._persist_transaction, payload)
        except Neo4jError as exc:
            raise KnowledgeGraphStoreError(
                "Unable to persist extraction result in Neo4j"
            ) from exc

        logger.info(
            "Persisted knowledge-graph extraction",
            extra={
                "document_id": result.source.document_id,
                "chunk_id": result.source.chunk_id,
                "entity_count": len(result.entities),
                "relationship_count": len(result.relationships),
            },
        )
        return GraphPersistenceResponse(
            document_id=result.source.document_id,
            chunk_id=result.source.chunk_id,
            entity_count=len(result.entities),
            mention_count=len(result.entities),
            relationship_count=len(result.relationships),
        )

    def get_entity_neighborhood(self, entity_id: str) -> EntityNeighborhood:
        """Return one entity and all directly connected semantic entities."""
        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(self._read_neighborhood, entity_id)
        except Neo4jError as exc:
            raise KnowledgeGraphStoreError(
                "Unable to query entity neighborhood from Neo4j"
            ) from exc

    def expand_seed_chunks(
        self,
        seed_chunk_ids: Sequence[str],
        hops: int,
        max_entities: int,
        max_relationships: int,
        max_supporting_chunks: int,
    ) -> GraphExpansionResult:
        """Expand vector seed chunks through a strictly bounded entity graph."""
        if hops not in (1, 2):
            raise KnowledgeGraphInputError("Graph expansion hops must be 1 or 2")
        if not seed_chunk_ids:
            return GraphExpansionResult()
        if min(max_entities, max_relationships, max_supporting_chunks) <= 0:
            raise KnowledgeGraphInputError("Graph expansion bounds must be positive")
        unique_chunk_ids = list(dict.fromkeys(seed_chunk_ids))
        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_read(
                    self._read_hybrid_expansion,
                    unique_chunk_ids,
                    hops,
                    max_entities,
                    max_relationships,
                    max_supporting_chunks,
                )
        except Neo4jError as exc:
            raise KnowledgeGraphStoreError(
                "Unable to expand hybrid retrieval context from Neo4j"
            ) from exc

    @staticmethod
    def _persist_transaction(transaction: Transaction, payload: dict[str, Any]) -> None:
        chunk = transaction.run(
            CHUNK_LOOKUP_CYPHER,
            document_id=payload["document_id"],
            chunk_id=payload["chunk_id"],
        ).single()
        if chunk is None:
            raise MissingChunkError(
                f"Chunk {payload['chunk_id']!r} is not linked to document "
                f"{payload['document_id']!r}"
            )

        transaction.run(ENTITY_UPSERT_CYPHER, entities=payload["entities"]).consume()
        transaction.run(
            MENTION_UPSERT_CYPHER,
            entities=payload["entities"],
            document_id=payload["document_id"],
            chunk_id=payload["chunk_id"],
            chunk_index=payload["chunk_index"],
            source=payload["source"],
            source_metadata_json=payload["source_metadata_json"],
        ).consume()
        for relationship in payload["relationships"]:
            relationship_type = relationship["relationship_type"]
            query = f"""
            MATCH (source:Entity {{entity_id: $source_entity}})
            MATCH (target:Entity {{entity_id: $target_entity}})
            MERGE (source)-[semantic:{relationship_type}]->(target)
            ON CREATE SET semantic.semantic_key = $semantic_key,
                          semantic.description = $description,
                          semantic.evidence_ids = [],
                          semantic.evidence_records = []
            SET semantic.description = CASE
                    WHEN semantic.description IS NULL OR trim(semantic.description) = ''
                    THEN $description
                    ELSE semantic.description
                END,
                semantic.evidence_records = CASE
                    WHEN $evidence_id IN coalesce(semantic.evidence_ids, [])
                    THEN coalesce(semantic.evidence_records, [])
                    ELSE coalesce(semantic.evidence_records, []) + $evidence_record
                END,
                semantic.evidence_ids = CASE
                    WHEN $evidence_id IN coalesce(semantic.evidence_ids, [])
                    THEN coalesce(semantic.evidence_ids, [])
                    ELSE coalesce(semantic.evidence_ids, []) + $evidence_id
                END
            """
            transaction.run(query, **relationship).consume()

    @staticmethod
    def _read_neighborhood(
        transaction: Transaction, entity_id: str
    ) -> EntityNeighborhood:
        entity_record = transaction.run(
            """
            MATCH (entity:Entity {entity_id: $entity_id})
            RETURN entity.entity_id AS entity_id,
                   entity.name AS name,
                   entity.normalized_name AS normalized_name,
                   entity.entity_type AS entity_type,
                   entity.description AS description
            """,
            entity_id=entity_id,
        ).single()
        if entity_record is None:
            raise EntityNotFoundError(f"Entity not found: {entity_id}")

        entity = Neo4jKnowledgeGraphStore._record_to_entity(entity_record)
        records = transaction.run(
            """
            MATCH (center:Entity {entity_id: $entity_id})
                  -[relationship]-(neighbor:Entity)
            RETURN startNode(relationship).entity_id AS source_entity_id,
                   startNode(relationship).name AS source_name,
                   startNode(relationship).normalized_name AS source_normalized_name,
                   startNode(relationship).entity_type AS source_entity_type,
                   startNode(relationship).description AS source_description,
                   endNode(relationship).entity_id AS target_entity_id,
                   endNode(relationship).name AS target_name,
                   endNode(relationship).normalized_name AS target_normalized_name,
                   endNode(relationship).entity_type AS target_entity_type,
                   endNode(relationship).description AS target_description,
                   type(relationship) AS relationship_type,
                   relationship.description AS relationship_description,
                   coalesce(relationship.evidence_records, []) AS evidence_records
            ORDER BY relationship_type, target_entity_id, source_entity_id
            """,
            entity_id=entity_id,
        )
        connections = [
            Neo4jKnowledgeGraphStore._record_to_connection(record)
            for record in records
        ]
        return EntityNeighborhood(entity=entity, connections=connections)

    @staticmethod
    def _read_hybrid_expansion(
        transaction: Transaction,
        seed_chunk_ids: list[str],
        hops: int,
        max_entities: int,
        max_relationships: int,
        max_supporting_chunks: int,
    ) -> GraphExpansionResult:
        seed_records = transaction.run(
            """
            MATCH (chunk:Chunk)-[:MENTIONS]->(entity:Entity)
            WHERE chunk.chunk_id IN $seed_chunk_ids
            RETURN entity.entity_id AS entity_id,
                   entity.name AS name,
                   entity.normalized_name AS normalized_name,
                   entity.entity_type AS entity_type,
                   entity.description AS description,
                   collect(DISTINCT chunk.chunk_id) AS seed_chunk_ids
            ORDER BY entity_id
            LIMIT $max_entities
            """,
            seed_chunk_ids=seed_chunk_ids,
            max_entities=max_entities,
        )
        entities: dict[str, GraphExpandedEntity] = {}
        for record in seed_records:
            graph_entity = Neo4jKnowledgeGraphStore._record_to_entity(record)
            entities[graph_entity.entity_id] = GraphExpandedEntity(
                entity=graph_entity,
                graph_distance=0,
                seed_chunk_ids=sorted(record["seed_chunk_ids"]),
            )
        if not entities:
            return GraphExpansionResult()

        # The only interpolated value is the integer validated above as exactly 1 or 2.
        path_query = f"""
            MATCH (seed:Entity)
            WHERE seed.entity_id IN $seed_entity_ids
            MATCH path = (seed)-[*1..{hops}]-(related:Entity)
            WHERE all(edge IN relationships(path) WHERE type(edge) <> 'MENTIONS')
              AND all(node IN nodes(path)
                      WHERE single(other IN nodes(path) WHERE other = node))
            WITH seed, path, relationships(path) AS semantic
            ORDER BY length(path), related.entity_id
            LIMIT $max_relationships
            UNWIND range(0, size(semantic) - 1) AS edge_index
            WITH seed, path, semantic[edge_index] AS edge, edge_index
            RETURN startNode(edge).entity_id AS source_entity_id,
                   startNode(edge).name AS source_name,
                   startNode(edge).normalized_name AS source_normalized_name,
                   startNode(edge).entity_type AS source_entity_type,
                   startNode(edge).description AS source_description,
                   endNode(edge).entity_id AS target_entity_id,
                   endNode(edge).name AS target_name,
                   endNode(edge).normalized_name AS target_normalized_name,
                   endNode(edge).entity_type AS target_entity_type,
                   endNode(edge).description AS target_description,
                   edge.semantic_key AS relationship_id,
                   type(edge) AS relationship_type,
                   edge.description AS relationship_description,
                   coalesce(edge.evidence_records, []) AS evidence_records,
                   edge_index + 1 AS graph_distance,
                   seed.entity_id AS seed_entity_id
            ORDER BY graph_distance, relationship_type, source_entity_id,
                     target_entity_id, relationship_id
            """
        path_records = transaction.run(
            path_query,
            seed_entity_ids=sorted(entities),
            max_relationships=max_relationships,
        )
        relationships: dict[str, GraphExpandedRelationship] = {}
        for record in path_records:
            distance = min(int(record["graph_distance"]), hops)
            for prefix in ("source_", "target_"):
                graph_entity = Neo4jKnowledgeGraphStore._record_to_entity(record, prefix)
                existing = entities.get(graph_entity.entity_id)
                candidate_distance = (
                    0 if graph_entity.entity_id in entities and existing.graph_distance == 0
                    else distance
                )
                if existing is None and len(entities) < max_entities:
                    entities[graph_entity.entity_id] = GraphExpandedEntity(
                        entity=graph_entity,
                        graph_distance=candidate_distance,
                        seed_chunk_ids=[],
                    )
                elif existing is not None and distance < existing.graph_distance:
                    existing.graph_distance = distance

            relationship_id = record["relationship_id"] or _semantic_key(
                record["source_entity_id"],
                record["relationship_type"],
                record["target_entity_id"],
            )
            evidence = Neo4jKnowledgeGraphStore._parse_evidence_records(
                record["evidence_records"]
            )
            candidate = GraphExpandedRelationship(
                relationship_id=relationship_id,
                source_entity_id=record["source_entity_id"],
                target_entity_id=record["target_entity_id"],
                relationship_type=record["relationship_type"],
                description=record["relationship_description"],
                graph_distance=distance,
                evidence=evidence,
            )
            if (
                candidate.source_entity_id not in entities
                or candidate.target_entity_id not in entities
            ):
                continue
            existing_relationship = relationships.get(relationship_id)
            if existing_relationship is None:
                relationships[relationship_id] = candidate
            else:
                existing_relationship.graph_distance = min(
                    existing_relationship.graph_distance, distance
                )
                known_evidence = {
                    item.evidence_id for item in existing_relationship.evidence
                }
                existing_relationship.evidence.extend(
                    item for item in evidence if item.evidence_id not in known_evidence
                )

        entity_ids = sorted(entities)[:max_entities]
        supporting_records = transaction.run(
            """
            MATCH (document:Document)-[:HAS_CHUNK]->(chunk:Chunk)
                  -[:MENTIONS]->(entity:Entity)
            WHERE entity.entity_id IN $entity_ids
            WITH document, chunk,
                 collect(DISTINCT entity.entity_id) AS entity_ids
            RETURN chunk.chunk_id AS chunk_id,
                   document.document_id AS document_id,
                   chunk.text AS text,
                   chunk.chunk_index AS chunk_index,
                   document.source AS source,
                   chunk.source_metadata_json AS source_metadata_json,
                   entity_ids
            ORDER BY chunk.chunk_id
            LIMIT $max_supporting_chunks
            """,
            entity_ids=entity_ids,
            max_supporting_chunks=max_supporting_chunks,
        )
        evidence_by_chunk: dict[str, set[str]] = {}
        for relationship in relationships.values():
            for evidence in relationship.evidence:
                evidence_by_chunk.setdefault(evidence.provenance.chunk_id, set()).add(
                    relationship.relationship_id
                )

        supporting_chunks: list[GraphSupportingChunk] = []
        for record in supporting_records:
            related_entity_ids = sorted(record["entity_ids"])
            entity_distances = [
                entities[entity_id].graph_distance
                for entity_id in related_entity_ids
                if entity_id in entities
            ]
            relationship_ids = sorted(evidence_by_chunk.get(record["chunk_id"], set()))
            sources = [RetrievalSource.GRAPH_ENTITY]
            if relationship_ids:
                sources.extend(
                    [
                        RetrievalSource.GRAPH_RELATIONSHIP,
                        RetrievalSource.RELATIONSHIP_EVIDENCE,
                    ]
                )
            supporting_chunks.append(
                GraphSupportingChunk(
                    chunk_id=record["chunk_id"],
                    document_id=record["document_id"],
                    text=record["text"],
                    chunk_index=record["chunk_index"],
                    source=record["source"],
                    source_metadata=_load_json_metadata(
                        record["source_metadata_json"]
                    ),
                    graph_distance=min(entity_distances, default=0),
                    entity_ids=related_entity_ids,
                    relationship_ids=relationship_ids,
                    retrieval_sources=sources,
                )
            )

        return GraphExpansionResult(
            entities=sorted(
                entities.values(),
                key=lambda item: (item.graph_distance, item.entity.entity_id),
            )[:max_entities],
            relationships=sorted(
                relationships.values(),
                key=lambda item: (
                    item.graph_distance,
                    item.relationship_type,
                    item.relationship_id,
                ),
            )[:max_relationships],
            supporting_chunks=supporting_chunks,
        )

    @staticmethod
    def _validate_result(result: ExtractionResult) -> None:
        entity_ids: set[str] = set()
        for entity in result.entities:
            if entity.provenance != result.source:
                raise KnowledgeGraphInputError(
                    f"Entity {entity.entity_id} provenance does not match extraction source"
                )
            if entity.entity_id in entity_ids:
                raise KnowledgeGraphInputError(
                    f"Duplicate entity ID in extraction: {entity.entity_id}"
                )
            try:
                expected_normalized_name = normalize_entity_name(entity.name)
            except GraphValidationError as exc:
                raise KnowledgeGraphInputError(str(exc)) from exc
            if entity.normalized_name != expected_normalized_name:
                raise KnowledgeGraphInputError(
                    f"Entity {entity.entity_id} has a noncanonical normalized name"
                )
            expected_entity_id = deterministic_entity_id(
                entity.entity_type, expected_normalized_name
            )
            if entity.entity_id != expected_entity_id:
                raise KnowledgeGraphInputError(
                    f"Entity ID does not match deterministic identity: {entity.entity_id}"
                )
            entity_ids.add(entity.entity_id)

        evidence_ids: set[str] = set()
        for relationship in result.relationships:
            if relationship.provenance != result.source:
                raise KnowledgeGraphInputError(
                    "Relationship provenance does not match extraction source"
                )
            if relationship.relationship_id in evidence_ids:
                raise KnowledgeGraphInputError(
                    "Duplicate relationship evidence ID in extraction: "
                    f"{relationship.relationship_id}"
                )
            evidence_ids.add(relationship.relationship_id)
            if (
                relationship.source_entity not in entity_ids
                or relationship.target_entity not in entity_ids
            ):
                raise MissingEntityEndpointError(
                    f"Relationship {relationship.relationship_id} references an entity "
                    "not present in the extraction result"
                )
            if relationship.source_entity == relationship.target_entity:
                raise KnowledgeGraphInputError(
                    "Self-referential semantic relationships are not supported"
                )
            Neo4jKnowledgeGraphStore._validate_relationship_type(
                relationship.relationship_type
            )
            expected_relationship_id = deterministic_relationship_id(
                relationship.source_entity,
                relationship.relationship_type,
                relationship.target_entity,
                result.source.chunk_id,
            )
            if relationship.relationship_id != expected_relationship_id:
                raise KnowledgeGraphInputError(
                    "Relationship ID does not match deterministic evidence identity"
                )

    @staticmethod
    def _validate_relationship_type(value: str) -> None:
        try:
            normalized = normalize_relationship_type(value)
        except GraphValidationError as exc:
            raise KnowledgeGraphInputError(str(exc)) from exc
        if normalized != value or len(value) > 128:
            raise KnowledgeGraphInputError(
                "Relationship type must be normalized UPPER_SNAKE_CASE"
            )

    @staticmethod
    def _persistence_payload(result: ExtractionResult) -> dict[str, Any]:
        source_metadata_json = _dump_json(result.source.source_metadata)
        entities = [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "normalized_name": entity.normalized_name,
                "entity_type": entity.entity_type.value,
                "description": entity.description,
            }
            for entity in result.entities
        ]
        relationships = []
        for relationship in result.relationships:
            evidence = RelationshipEvidence(
                evidence_id=relationship.relationship_id,
                description=relationship.description,
                evidence=relationship.evidence,
                provenance=relationship.provenance,
            )
            relationships.append(
                {
                    "source_entity": relationship.source_entity,
                    "target_entity": relationship.target_entity,
                    "relationship_type": relationship.relationship_type,
                    "semantic_key": _semantic_key(
                        relationship.source_entity,
                        relationship.relationship_type,
                        relationship.target_entity,
                    ),
                    "description": relationship.description,
                    "evidence_id": relationship.relationship_id,
                    "evidence_record": _dump_json(
                        evidence.model_dump(mode="json")
                    ),
                }
            )
        return {
            "document_id": result.source.document_id,
            "chunk_id": result.source.chunk_id,
            "chunk_index": result.source.chunk_index,
            "source": result.source.source,
            "source_metadata_json": source_metadata_json,
            "entities": entities,
            "relationships": relationships,
        }

    @staticmethod
    def _record_to_entity(record: Any, prefix: str = "") -> GraphEntity:
        return GraphEntity(
            entity_id=record[f"{prefix}entity_id"],
            name=record[f"{prefix}name"],
            normalized_name=record[f"{prefix}normalized_name"],
            entity_type=record[f"{prefix}entity_type"],
            description=record[f"{prefix}description"],
        )

    @staticmethod
    def _record_to_connection(record: Any) -> GraphConnection:
        evidence = Neo4jKnowledgeGraphStore._parse_evidence_records(
            record["evidence_records"]
        )
        return GraphConnection(
            source_entity=Neo4jKnowledgeGraphStore._record_to_entity(
                record, "source_"
            ),
            target_entity=Neo4jKnowledgeGraphStore._record_to_entity(
                record, "target_"
            ),
            relationship_type=record["relationship_type"],
            description=record["relationship_description"],
            evidence=evidence,
        )

    @staticmethod
    def _parse_evidence_records(records: Sequence[str]) -> list[RelationshipEvidence]:
        evidence: list[RelationshipEvidence] = []
        evidence_ids: set[str] = set()
        for raw_record in records or []:
            try:
                parsed = RelationshipEvidence.model_validate_json(raw_record)
            except (ValueError, TypeError) as exc:
                raise KnowledgeGraphStoreError(
                    "Stored relationship evidence is not valid JSON"
                ) from exc
            if parsed.evidence_id not in evidence_ids:
                evidence.append(parsed)
                evidence_ids.add(parsed.evidence_id)
        return evidence


def _semantic_key(source_entity: str, relationship_type: str, target_entity: str) -> str:
    payload = f"{source_entity}\0{relationship_type}\0{target_entity}".encode("utf-8")
    return f"semantic_{hashlib.sha256(payload).hexdigest()}"


def _dump_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise KnowledgeGraphInputError("Graph provenance must be JSON serializable") from exc


def _load_json_metadata(value: str | None) -> dict[str, Any]:
    try:
        metadata = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError) as exc:
        raise KnowledgeGraphStoreError(
            "Stored chunk metadata is not valid JSON"
        ) from exc
    if not isinstance(metadata, dict):
        raise KnowledgeGraphStoreError("Stored chunk metadata must be a JSON object")
    return metadata
