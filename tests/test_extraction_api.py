"""API tests for extraction-only graph candidates."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_graph_extraction_service
from app.extraction.service import GraphExtractionService
from app.main import app
from app.models.graph import ProviderEntity, ProviderExtraction, ProviderRelationship


class ApiExtractionProvider:
    def extract(self, text: str) -> ProviderExtraction:
        return ProviderExtraction(
            entities=[
                ProviderEntity(
                    name="Information Security Team",
                    entity_type="BUSINESS_UNIT",
                ),
                ProviderEntity(
                    name="Identity Access System",
                    entity_type="SYSTEM",
                ),
            ],
            relationships=[
                ProviderRelationship(
                    source_entity="Information Security Team",
                    target_entity="Identity Access System",
                    relationship_type="MANAGES",
                    evidence=text,
                )
            ],
        )


@pytest.fixture
def extraction_client() -> Iterator[TestClient]:
    app.dependency_overrides[get_graph_extraction_service] = lambda: (
        GraphExtractionService(ApiExtractionProvider())
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_graph_extraction_api_response_structure(
    extraction_client: TestClient,
) -> None:
    response = extraction_client.post(
        "/extract/graph",
        json={
            "text": (
                "The Information Security Team manages the Identity Access System."
            ),
            "document_id": "doc_api",
            "chunk_id": "chunk_api",
            "chunk_index": 4,
            "source_metadata": {"filename": "policy.md"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == {
        "document_id": "doc_api",
        "chunk_id": "chunk_api",
        "source": "policy.md",
        "chunk_index": 4,
        "source_metadata": {"filename": "policy.md"},
    }
    assert [entity["entity_type"] for entity in payload["entities"]] == [
        "BUSINESS_UNIT",
        "SYSTEM",
    ]
    assert payload["relationships"][0]["relationship_type"] == "MANAGES"
    assert payload["relationships"][0]["evidence"].startswith("The Information")


def test_graph_extraction_api_generates_deterministic_source_ids(
    extraction_client: TestClient,
) -> None:
    request = {"text": "The Information Security Team manages a system."}

    first = extraction_client.post("/extract/graph", json=request).json()
    second = extraction_client.post("/extract/graph", json=request).json()

    assert first["source"]["document_id"] == second["source"]["document_id"]
    assert first["source"]["chunk_id"] == second["source"]["chunk_id"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": "valid", "chunk_index": -1},
        {"text": "valid", "source_metadata": "not-an-object"},
    ],
)
def test_graph_extraction_api_request_validation(
    extraction_client: TestClient, payload: dict[str, object]
) -> None:
    response = extraction_client.post("/extract/graph", json=payload)

    assert response.status_code == 422
