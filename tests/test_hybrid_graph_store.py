"""Tests for safe, bounded Neo4j graph expansion."""

import json
from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import Neo4jError

from app.graph.exceptions import (
    KnowledgeGraphInputError,
    KnowledgeGraphStoreError,
)
from app.graph.knowledge_store import Neo4jKnowledgeGraphStore


def _entity_record(entity_id: str, seed_chunks: list[str] | None = None) -> dict:
    record = {
        "entity_id": entity_id,
        "name": entity_id.replace("_", " ").title(),
        "normalized_name": entity_id.replace("_", " "),
        "entity_type": "SYSTEM",
        "description": None,
    }
    if seed_chunks is not None:
        record["seed_chunk_ids"] = seed_chunks
    return record


def _evidence() -> str:
    return json.dumps(
        {
            "evidence_id": "evidence_1",
            "description": "Policy scope",
            "evidence": "The policy governs the identity system.",
            "provenance": {
                "document_id": "doc_2",
                "chunk_id": "chunk_support",
                "source": "/data/standard.md",
                "chunk_index": 3,
                "source_metadata": {"filename": "standard.md"},
            },
        }
    )


def _relationship_record(distance: int = 1) -> dict:
    source = _entity_record("entity_policy")
    target = _entity_record("entity_system")
    return {
        **{f"source_{key}": value for key, value in source.items()},
        **{f"target_{key}": value for key, value in target.items()},
        "relationship_id": "semantic_1",
        "relationship_type": "GOVERNS",
        "relationship_description": "Policy scope",
        "evidence_records": [_evidence()],
        "graph_distance": distance,
        "seed_entity_id": "entity_policy",
    }


def _support_record() -> dict:
    return {
        "chunk_id": "chunk_support",
        "document_id": "doc_2",
        "text": "The policy governs the identity system.",
        "chunk_index": 3,
        "source": "/data/standard.md",
        "source_metadata_json": '{"filename":"standard.md","page":2}',
        "entity_ids": ["entity_system", "entity_policy"],
    }


def _transaction(
    path_records: list[dict] | None = None,
    support_records: list[dict] | None = None,
) -> MagicMock:
    transaction = MagicMock()
    transaction.run.side_effect = [
        [_entity_record("entity_policy", ["chunk_seed"])],
        path_records if path_records is not None else [_relationship_record()],
        support_records if support_records is not None else [_support_record()],
    ]
    return transaction


def test_one_hop_expansion_returns_entities_relationships_and_supporting_chunks() -> None:
    transaction = _transaction()

    result = Neo4jKnowledgeGraphStore._read_hybrid_expansion(
        transaction, ["chunk_seed"], 1, 10, 10, 10
    )

    assert [item.entity.entity_id for item in result.entities] == [
        "entity_policy",
        "entity_system",
    ]
    assert result.relationships[0].relationship_type == "GOVERNS"
    assert result.relationships[0].evidence[0].provenance.chunk_id == (
        "chunk_support"
    )
    assert result.supporting_chunks[0].relationship_ids == ["semantic_1"]
    assert result.supporting_chunks[0].source_metadata["page"] == 2


def test_two_hop_expansion_uses_only_validated_literal_depth() -> None:
    transaction = _transaction(path_records=[_relationship_record(distance=2)])

    result = Neo4jKnowledgeGraphStore._read_hybrid_expansion(
        transaction, ["chunk_seed"], 2, 10, 10, 10
    )

    path_query = transaction.run.call_args_list[1].args[0]
    assert "[*1..2]" in path_query
    assert result.relationships[0].graph_distance == 2


def test_cycle_and_duplicate_path_rows_are_deduplicated() -> None:
    record = _relationship_record()
    transaction = _transaction(path_records=[record, record])

    result = Neo4jKnowledgeGraphStore._read_hybrid_expansion(
        transaction, ["chunk_seed"], 2, 10, 10, 10
    )

    path_query = transaction.run.call_args_list[1].args[0]
    assert "single(other IN nodes(path)" in path_query
    assert len(result.relationships) == 1
    assert len(result.relationships[0].evidence) == 1
    assert len(result.entities) == 2


def test_graph_queries_parameterize_ids_and_bounds() -> None:
    transaction = _transaction()
    malicious_id = "x') MATCH (n) DETACH DELETE n //"

    Neo4jKnowledgeGraphStore._read_hybrid_expansion(
        transaction, [malicious_id], 1, 7, 8, 9
    )

    seed_query = transaction.run.call_args_list[0].args[0]
    assert malicious_id not in seed_query
    assert transaction.run.call_args_list[0].kwargs == {
        "seed_chunk_ids": [malicious_id],
        "max_entities": 7,
    }
    assert transaction.run.call_args_list[1].kwargs["max_relationships"] == 8
    assert transaction.run.call_args_list[2].kwargs["max_supporting_chunks"] == 9


def test_final_graph_result_respects_global_entity_and_relationship_caps() -> None:
    first = _relationship_record()
    second = _relationship_record()
    second["relationship_id"] = "semantic_2"
    second["relationship_type"] = "DEPENDS_ON"
    transaction = _transaction(path_records=[first, second])

    result = Neo4jKnowledgeGraphStore._read_hybrid_expansion(
        transaction, ["chunk_seed"], 2, 2, 1, 1
    )

    assert len(result.entities) <= 2
    assert len(result.relationships) <= 1
    assert len(result.supporting_chunks) <= 1


def test_no_seed_entities_returns_empty_result_without_more_queries() -> None:
    transaction = MagicMock()
    transaction.run.return_value = []

    result = Neo4jKnowledgeGraphStore._read_hybrid_expansion(
        transaction, ["chunk_seed"], 1, 10, 10, 10
    )

    assert result.entities == []
    assert result.relationships == []
    assert result.supporting_chunks == []
    assert transaction.run.call_count == 1


@pytest.mark.parametrize("hops", [0, 3, 999])
def test_store_rejects_unsafe_hops_before_database_access(hops: int) -> None:
    driver = MagicMock()
    store = Neo4jKnowledgeGraphStore(driver, "neo4j")

    with pytest.raises(KnowledgeGraphInputError, match="1 or 2"):
        store.expand_seed_chunks(["chunk_seed"], hops, 10, 10, 10)

    driver.session.assert_not_called()


def test_empty_seed_chunks_short_circuit_without_database_access() -> None:
    driver = MagicMock()
    store = Neo4jKnowledgeGraphStore(driver, "neo4j")

    assert store.expand_seed_chunks([], 1, 10, 10, 10).supporting_chunks == []
    driver.session.assert_not_called()


def test_neo4j_expansion_failure_is_wrapped() -> None:
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.execute_read.side_effect = Neo4jError("offline")
    store = Neo4jKnowledgeGraphStore(driver, "neo4j")

    with pytest.raises(KnowledgeGraphStoreError, match="expand"):
        store.expand_seed_chunks(["chunk_seed"], 1, 10, 10, 10)


def test_malformed_relationship_evidence_is_reported() -> None:
    record = _relationship_record()
    record["evidence_records"] = ["not-json"]
    transaction = _transaction(path_records=[record])

    with pytest.raises(KnowledgeGraphStoreError, match="evidence"):
        Neo4jKnowledgeGraphStore._read_hybrid_expansion(
            transaction, ["chunk_seed"], 1, 10, 10, 10
        )
