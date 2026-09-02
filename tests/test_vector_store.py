"""Unit tests for Neo4j chunk persistence and vector indexes."""

from unittest.mock import MagicMock

import pytest

from app.graph.exceptions import VectorIndexConfigurationError, VectorStoreError
from app.graph.vector_store import Neo4jVectorStore, VectorIndexConfig
from app.models.document import Document
from app.models.vector import EmbeddedChunk


def _document() -> Document:
    return Document(
        document_id="doc_1",
        source="/data/policy.md",
        content="Retention policy",
        metadata={"filename": "policy.md", "department": "Legal"},
    )


def _embedded_chunk(document_id: str = "doc_1") -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id="chunk_1",
        document_id=document_id,
        text="Records are retained for seven years.",
        chunk_index=0,
        source_metadata={"filename": "policy.md", "page": 2},
        embedding=[0.1, 0.2, 0.3],
    )


def _store(session: MagicMock | None = None) -> tuple[Neo4jVectorStore, MagicMock]:
    active_session = session or MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = active_session
    store = Neo4jVectorStore(
        driver=driver,
        database="neo4j",
        index_config=VectorIndexConfig("chunk_embedding_index", 3, "cosine"),
    )
    return store, active_session


def test_vector_index_is_created_only_when_missing() -> None:
    store, session = _store()
    results = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    results[2].single.return_value = None
    session.run.side_effect = results

    store.ensure_schema()

    assert session.run.call_count == 4
    create_query = session.run.call_args_list[3].args[0]
    assert "CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS" in create_query
    assert "`vector.dimensions`: 3" in create_query
    assert "'cosine'" in create_query


def test_existing_compatible_vector_index_is_not_recreated() -> None:
    store, session = _store()
    results = [MagicMock(), MagicMock(), MagicMock()]
    results[2].single.return_value = {
        "options": {
            "indexConfig": {
                "vector.dimensions": 3,
                "vector.similarity_function": "COSINE",
            }
        }
    }
    session.run.side_effect = results

    store.ensure_schema()

    assert session.run.call_count == 3


def test_existing_incompatible_vector_index_fails_clearly() -> None:
    store, session = _store()
    results = [MagicMock(), MagicMock(), MagicMock()]
    results[2].single.return_value = {
        "options": {
            "indexConfig": {
                "vector.dimensions": 1_536,
                "vector.similarity_function": "cosine",
            }
        }
    }
    session.run.side_effect = results

    with pytest.raises(VectorIndexConfigurationError, match="dimension=1536"):
        store.ensure_schema()


def test_chunk_upsert_uses_merge_and_preserves_metadata() -> None:
    store, session = _store()
    transaction = MagicMock()
    session.execute_write.side_effect = lambda callback, params: callback(
        transaction, params
    )

    store.upsert_document_chunks(_document(), [_embedded_chunk()])
    store.upsert_document_chunks(_document(), [_embedded_chunk()])

    assert session.execute_write.call_count == 2
    query = transaction.run.call_args.args[0]
    parameters = transaction.run.call_args.kwargs
    assert "MERGE (document:Document" in query
    assert "MERGE (chunk:Chunk" in query
    assert "MERGE (document)-[:HAS_CHUNK]->(chunk)" in query
    assert parameters["chunks"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert '"department":"Legal"' in parameters["document"]["metadata_json"]
    assert '"filename":"policy.md"' in (
        parameters["chunks"][0]["source_metadata_json"]
    )


def test_chunk_upsert_rejects_wrong_document_reference() -> None:
    store, _ = _store()

    with pytest.raises(VectorStoreError, match="must reference"):
        store.upsert_document_chunks(_document(), [_embedded_chunk("doc_other")])


def test_vector_search_returns_structured_results_and_metadata() -> None:
    store, session = _store()
    session.run.return_value = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "text": "Records are retained for seven years.",
            "score": 0.92,
            "chunk_index": 2,
            "source": "/data/policy.md",
            "source_metadata_json": '{"filename":"policy.md","page":2}',
        }
    ]

    results = store.vector_search([0.1, 0.2, 0.3], top_k=4)

    assert results[0].score == 0.92
    assert results[0].source_metadata == {"filename": "policy.md", "page": 2}
    assert results[0].chunk_index == 2
    assert results[0].source == "/data/policy.md"
    assert session.run.call_args.kwargs["top_k"] == 4
    assert session.run.call_args.kwargs["index_name"] == "chunk_embedding_index"


def test_vector_search_rejects_incompatible_query_dimension() -> None:
    store, _ = _store()

    with pytest.raises(VectorIndexConfigurationError, match="index expects 3"):
        store.vector_search([0.1, 0.2], top_k=5)
