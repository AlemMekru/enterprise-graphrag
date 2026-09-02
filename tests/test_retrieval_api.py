"""API tests for the vector retrieval endpoint."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_vector_retriever
from app.main import app
from app.models.vector import VectorRetrievalResult


class StubRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[VectorRetrievalResult]:
        self.calls.append((query, top_k))
        return [
            VectorRetrievalResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                text="Records are retained for seven years.",
                score=0.91,
                source_metadata={"filename": "information-security-policy.md"},
            )
        ]


@pytest.fixture
def retrieval_client() -> Iterator[tuple[TestClient, StubRetriever]]:
    retriever = StubRetriever()
    app.dependency_overrides[get_vector_retriever] = lambda: retriever
    with TestClient(app) as client:
        yield client, retriever
    app.dependency_overrides.clear()


def test_vector_retrieval_response_is_structured(
    retrieval_client: tuple[TestClient, StubRetriever],
) -> None:
    client, retriever = retrieval_client

    response = client.post(
        "/retrieve/vector",
        json={"query": "What is the data retention policy?", "top_k": 3},
    )

    assert response.status_code == 200
    assert retriever.calls == [("What is the data retention policy?", 3)]
    assert response.json() == {
        "query": "What is the data retention policy?",
        "results": [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "text": "Records are retained for seven years.",
                "score": 0.91,
                "chunk_index": 0,
                "source": None,
                "source_metadata": {
                    "filename": "information-security-policy.md"
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "top_k": 3},
        {"query": "policy", "top_k": 0},
        {"query": "policy", "top_k": 101},
    ],
)
def test_vector_retrieval_request_validation(
    retrieval_client: tuple[TestClient, StubRetriever], payload: dict[str, object]
) -> None:
    client, _ = retrieval_client

    response = client.post("/retrieve/vector", json=payload)

    assert response.status_code == 422
