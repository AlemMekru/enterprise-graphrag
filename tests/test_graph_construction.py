"""Tests for Phase 4 graph construction orchestration."""

from unittest.mock import MagicMock

from app.graph.construction import GraphConstructionService
from app.models.graph import ExtractionResult, GraphPersistenceResponse
from tests.test_knowledge_store import _extraction


def test_graph_construction_persists_one_result() -> None:
    store = MagicMock()
    result = _extraction()
    summary = GraphPersistenceResponse(
        document_id="doc_1",
        chunk_id="chunk_1",
        entity_count=2,
        mention_count=2,
        relationship_count=1,
    )
    store.persist_extraction_result.return_value = summary

    response = GraphConstructionService(store).persist_result(result)

    assert response == summary
    store.ensure_schema.assert_called_once_with()
    store.persist_extraction_result.assert_called_once_with(result)


def test_graph_construction_initializes_schema_once_for_multiple_results() -> None:
    store = MagicMock()
    results: list[ExtractionResult] = [_extraction("chunk_1"), _extraction("chunk_2")]
    store.persist_extraction_result.side_effect = [MagicMock(), MagicMock()]

    responses = GraphConstructionService(store).persist_results(results)

    assert len(responses) == 2
    store.ensure_schema.assert_called_once_with()
    assert store.persist_extraction_result.call_count == 2
