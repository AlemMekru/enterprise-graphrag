"""Unit tests for transactional Neo4j knowledge-graph persistence."""

import json
from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import Neo4jError

from app.extraction.normalization import (
    deterministic_entity_id,
    deterministic_relationship_id,
)
from app.graph.exceptions import (
    EntityNotFoundError,
    KnowledgeGraphInputError,
    KnowledgeGraphStoreError,
    MissingChunkError,
    MissingEntityEndpointError,
)
from app.graph.knowledge_store import Neo4jKnowledgeGraphStore
from app.models.graph import (
    Entity,
    EntityType,
    ExtractionResult,
    GraphProvenance,
    Relationship,
)


def _provenance(chunk_id: str = "chunk_1") -> GraphProvenance:
    return GraphProvenance(
        document_id="doc_1",
        chunk_id=chunk_id,
        source="/data/policy.md",
        chunk_index=1,
        source_metadata={"filename": "policy.md", "department": "Security"},
    )


def _extraction(chunk_id: str = "chunk_1") -> ExtractionResult:
    provenance = _provenance(chunk_id)
    source_id = deterministic_entity_id(EntityType.BUSINESS_UNIT, "security team")
    target_id = deterministic_entity_id(EntityType.SYSTEM, "identity access system")
    entities = [
        Entity(
            entity_id=source_id,
            name="Security Team",
            normalized_name="security team",
            entity_type=EntityType.BUSINESS_UNIT,
            description="Security function",
            provenance=provenance,
        ),
        Entity(
            entity_id=target_id,
            name="Identity Access System",
            normalized_name="identity access system",
            entity_type=EntityType.SYSTEM,
            provenance=provenance,
        ),
    ]
    relationship_type = "MANAGES"
    relationship = Relationship(
        relationship_id=deterministic_relationship_id(
            source_id, relationship_type, target_id, chunk_id
        ),
        source_entity=source_id,
        target_entity=target_id,
        relationship_type=relationship_type,
        description="Management responsibility",
        evidence="Security Team manages the Identity Access System.",
        provenance=provenance,
    )
    return ExtractionResult(
        entities=entities,
        relationships=[relationship],
        source=provenance,
    )


def _store(session: MagicMock | None = None) -> tuple[Neo4jKnowledgeGraphStore, MagicMock]:
    active_session = session or MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = active_session
    return Neo4jKnowledgeGraphStore(driver, "neo4j"), active_session


def _execute_transactions(session: MagicMock, transaction: MagicMock) -> None:
    session.execute_write.side_effect = lambda callback, payload: callback(
        transaction, payload
    )


def test_schema_initialization_creates_shared_uniqueness_constraints() -> None:
    store, session = _store()

    store.ensure_schema()
    store.ensure_schema()

    assert session.run.call_count == 6
    queries = [call.args[0] for call in session.run.call_args_list[:3]]
    assert any("Document" in query and "document_id" in query for query in queries)
    assert any("Chunk" in query and "chunk_id" in query for query in queries)
    assert any("Entity" in query and "entity_id" in query for query in queries)
    assert all("IF NOT EXISTS" in query for query in queries)


def test_schema_initialization_failure_is_wrapped() -> None:
    store, session = _store()
    session.run.side_effect = Neo4jError("schema unavailable")

    with pytest.raises(KnowledgeGraphStoreError, match="initialize"):
        store.ensure_schema()


def test_persistence_uses_one_transaction_and_idempotent_merge_queries() -> None:
    store, session = _store()
    transaction = MagicMock()
    _execute_transactions(session, transaction)

    summary = store.persist_extraction_result(_extraction())

    session.execute_write.assert_called_once()
    assert summary.entity_count == 2
    assert summary.mention_count == 2
    assert summary.relationship_count == 1
    queries = [call.args[0] for call in transaction.run.call_args_list]
    assert "HAS_CHUNK" in queries[0]
    assert "MERGE (entity:Entity" in queries[1]
    assert "MERGE (chunk)-[mention:MENTIONS]->(entity)" in queries[2]
    assert "MERGE (source)-[semantic:MANAGES]->(target)" in queries[3]


def test_entity_upsert_does_not_overwrite_useful_description_with_empty_value() -> None:
    store, session = _store()
    transaction = MagicMock()
    _execute_transactions(session, transaction)

    store.persist_extraction_result(_extraction())

    entity_query = transaction.run.call_args_list[1].args[0]
    assert "WHEN entity.description IS NULL" in entity_query
    assert "ELSE entity.description" in entity_query


def test_repeated_persistence_uses_stable_entities_mentions_and_evidence_ids() -> None:
    store, session = _store()
    transaction = MagicMock()
    _execute_transactions(session, transaction)
    extraction = _extraction()

    store.persist_extraction_result(extraction)
    first_relationship_parameters = transaction.run.call_args.kwargs
    store.persist_extraction_result(extraction)
    second_relationship_parameters = transaction.run.call_args.kwargs

    assert session.execute_write.call_count == 2
    assert first_relationship_parameters["evidence_id"] == (
        second_relationship_parameters["evidence_id"]
    )
    assert first_relationship_parameters["semantic_key"] == (
        second_relationship_parameters["semantic_key"]
    )
    relationship_query = transaction.run.call_args.args[0]
    assert "$evidence_id IN coalesce(semantic.evidence_ids, [])" in relationship_query


def test_relationship_evidence_preserves_full_provenance() -> None:
    store, session = _store()
    transaction = MagicMock()
    _execute_transactions(session, transaction)

    store.persist_extraction_result(_extraction())

    parameters = transaction.run.call_args.kwargs
    evidence_record = json.loads(parameters["evidence_record"])
    assert evidence_record["evidence_id"] == parameters["evidence_id"]
    assert evidence_record["provenance"]["document_id"] == "doc_1"
    assert evidence_record["provenance"]["chunk_id"] == "chunk_1"
    assert evidence_record["provenance"]["source_metadata"]["filename"] == (
        "policy.md"
    )


def test_distinct_chunks_create_distinct_relationship_evidence() -> None:
    first = Neo4jKnowledgeGraphStore._persistence_payload(_extraction("chunk_1"))
    second = Neo4jKnowledgeGraphStore._persistence_payload(_extraction("chunk_2"))

    assert first["entities"][0]["entity_id"] == second["entities"][0]["entity_id"]
    assert first["relationships"][0]["semantic_key"] == (
        second["relationships"][0]["semantic_key"]
    )
    assert first["relationships"][0]["evidence_id"] != (
        second["relationships"][0]["evidence_id"]
    )


def test_missing_chunk_aborts_transaction() -> None:
    store, session = _store()
    transaction = MagicMock()
    transaction.run.return_value.single.return_value = None
    _execute_transactions(session, transaction)

    with pytest.raises(MissingChunkError, match="not linked"):
        store.persist_extraction_result(_extraction())

    assert transaction.run.call_count == 1


def test_missing_relationship_endpoint_is_rejected_before_database_access() -> None:
    store, session = _store()
    extraction = _extraction()
    extraction.relationships[0].target_entity = "entity_missing"

    with pytest.raises(MissingEntityEndpointError):
        store.persist_extraction_result(extraction)

    session.execute_write.assert_not_called()


def test_noncanonical_entity_identity_is_rejected_before_database_access() -> None:
    store, session = _store()
    extraction = _extraction()
    extraction.entities[0].entity_id = "entity_forged"

    with pytest.raises(KnowledgeGraphInputError, match="deterministic identity"):
        store.persist_extraction_result(extraction)

    session.execute_write.assert_not_called()


def test_noncanonical_normalized_name_is_rejected() -> None:
    store, session = _store()
    extraction = _extraction()
    extraction.entities[0].normalized_name = "not the normalized display name"

    with pytest.raises(KnowledgeGraphInputError, match="noncanonical"):
        store.persist_extraction_result(extraction)

    session.execute_write.assert_not_called()


def test_blank_entity_name_is_rejected_as_graph_input() -> None:
    store, session = _store()
    extraction = _extraction()
    extraction.entities[0].name = "   "

    with pytest.raises(KnowledgeGraphInputError, match="must not be empty"):
        store.persist_extraction_result(extraction)

    session.execute_write.assert_not_called()


def test_noncanonical_relationship_evidence_id_is_rejected() -> None:
    store, session = _store()
    extraction = _extraction()
    extraction.relationships[0].relationship_id = "relationship_forged"

    with pytest.raises(KnowledgeGraphInputError, match="evidence identity"):
        store.persist_extraction_result(extraction)

    session.execute_write.assert_not_called()


@pytest.mark.parametrize(
    "relationship_type",
    ["manages", "MANAGES`) MATCH (node) DETACH DELETE node //", "HAS-DASH"],
)
def test_unsafe_or_noncanonical_relationship_type_is_rejected(
    relationship_type: str,
) -> None:
    store, session = _store()
    extraction = _extraction()
    extraction.relationships[0].relationship_type = relationship_type

    with pytest.raises(KnowledgeGraphInputError, match="Relationship type|Invalid"):
        store.persist_extraction_result(extraction)

    session.execute_write.assert_not_called()


def test_neo4j_transaction_failure_is_wrapped() -> None:
    store, session = _store()
    session.execute_write.side_effect = Neo4jError("database unavailable")

    with pytest.raises(KnowledgeGraphStoreError, match="Unable to persist"):
        store.persist_extraction_result(_extraction())


def test_neighborhood_returns_directed_connections_and_evidence() -> None:
    store, session = _store()
    transaction = MagicMock()
    session.execute_read.side_effect = lambda callback, entity_id: callback(
        transaction, entity_id
    )
    entity_result = MagicMock()
    entity_result.single.return_value = {
        "entity_id": "entity_source",
        "name": "Security Team",
        "normalized_name": "security team",
        "entity_type": "BUSINESS_UNIT",
        "description": "Security function",
    }
    connections_result = [
        {
            "source_entity_id": "entity_source",
            "source_name": "Security Team",
            "source_normalized_name": "security team",
            "source_entity_type": "BUSINESS_UNIT",
            "source_description": "Security function",
            "target_entity_id": "entity_target",
            "target_name": "Identity Access System",
            "target_normalized_name": "identity access system",
            "target_entity_type": "SYSTEM",
            "target_description": None,
            "relationship_type": "MANAGES",
            "relationship_description": "Management responsibility",
            "evidence_records": [
                json.dumps(
                    {
                        "evidence_id": "relationship_1",
                        "description": None,
                        "evidence": "Security Team manages the system.",
                        "provenance": _provenance().model_dump(mode="json"),
                    }
                )
            ],
        }
    ]
    transaction.run.side_effect = [entity_result, connections_result]

    neighborhood = store.get_entity_neighborhood("entity_source")

    assert neighborhood.entity.name == "Security Team"
    assert neighborhood.connections[0].relationship_type == "MANAGES"
    assert neighborhood.connections[0].target_entity.entity_type is EntityType.SYSTEM
    assert neighborhood.connections[0].evidence[0].provenance.chunk_id == "chunk_1"


def test_neighborhood_missing_entity_is_reported() -> None:
    store, session = _store()
    transaction = MagicMock()
    session.execute_read.side_effect = lambda callback, entity_id: callback(
        transaction, entity_id
    )
    transaction.run.return_value.single.return_value = None

    with pytest.raises(EntityNotFoundError):
        store.get_entity_neighborhood("entity_missing")
