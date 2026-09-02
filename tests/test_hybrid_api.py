"""API contract tests for retrieval-only hybrid GraphRAG context."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_hybrid_retriever
from app.graph.exceptions import KnowledgeGraphStoreError
from app.main import app
from app.models.hybrid import HybridRetrievalResult
from app.models.vector import VectorRetrievalResult


class StubHybridRetriever:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, int | None, int]] = []
        self.fail = fail

    def retrieve(
        self, query: str, top_k: int | None = None, graph_hops: int = 1
    ) -> HybridRetrievalResult:
        self.calls.append((query, top_k, graph_hops))
        if self.fail:
            raise KnowledgeGraphStoreError("Neo4j unavailable")
        vector = VectorRetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            text="The policy governs the system.",
            score=0.9,
            chunk_index=2,
            source="/data/policy.md",
            source_metadata={"filename": "policy.md"},
        )
        return HybridRetrievalResult(
            query=query,
            vector_seed_results=[vector],
            entities=[],
            relationships=[],
            context=[],
            graph_evidence_found=False,
        )


@pytest.fixture
def hybrid_client() -> Iterator[tuple[TestClient, StubHybridRetriever]]:
    retriever = StubHybridRetriever()
    app.dependency_overrides[get_hybrid_retriever] = lambda: retriever
    with TestClient(app) as client:
        yield client, retriever
    app.dependency_overrides.clear()


def test_hybrid_endpoint_returns_structured_context_only(
    hybrid_client: tuple[TestClient, StubHybridRetriever],
) -> None:
    client, retriever = hybrid_client

    response = client.post(
        "/retrieve/hybrid",
        json={"query": "Which systems are governed?", "top_k": 4, "graph_hops": 2},
    )

    assert response.status_code == 200
    assert retriever.calls == [("Which systems are governed?", 4, 2)]
    body = response.json()
    assert set(body) == {
        "query",
        "vector_seed_results",
        "entities",
        "relationships",
        "context",
        "graph_evidence_found",
    }
    assert "answer" not in body
    assert body["vector_seed_results"][0]["chunk_index"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "graph_hops": 1},
        {"query": "systems", "top_k": 0},
        {"query": "systems", "top_k": 101},
        {"query": "systems", "graph_hops": 0},
        {"query": "systems", "graph_hops": 3},
    ],
)
def test_hybrid_endpoint_rejects_unsafe_bounds(
    hybrid_client: tuple[TestClient, StubHybridRetriever],
    payload: dict[str, object],
) -> None:
    client, retriever = hybrid_client

    response = client.post("/retrieve/hybrid", json=payload)

    assert response.status_code == 422
    assert retriever.calls == []


def test_hybrid_endpoint_reports_database_failure_as_service_error() -> None:
    retriever = StubHybridRetriever(fail=True)
    app.dependency_overrides[get_hybrid_retriever] = lambda: retriever
    try:
        with TestClient(app) as client:
            response = client.post(
                "/retrieve/hybrid", json={"query": "systems"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Hybrid retrieval is temporarily unavailable"
    )
