"""API tests for knowledge-graph persistence and neighborhood queries."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_graph_construction_service,
    get_knowledge_graph_store,
)
from app.graph.exceptions import EntityNotFoundError, MissingChunkError
from app.main import app
from app.models.graph import (
    EntityNeighborhood,
    EntityType,
    GraphConnection,
    GraphEntity,
    GraphPersistenceResponse,
    RelationshipEvidence,
)
from tests.test_knowledge_store import _extraction

ENTITY_ID = "entity_" + "a" * 64
TARGET_ID = "entity_" + "b" * 64


class StubConstructionService:
    def persist_result(self, result: object) -> GraphPersistenceResponse:
        return GraphPersistenceResponse(
            document_id="doc_1",
            chunk_id="chunk_1",
            entity_count=2,
            mention_count=2,
            relationship_count=1,
        )


class StubKnowledgeGraphStore:
    def get_entity_neighborhood(self, entity_id: str) -> EntityNeighborhood:
        extraction = _extraction()
        source = GraphEntity(
            entity_id=ENTITY_ID,
            name="Security Team",
            normalized_name="security team",
            entity_type=EntityType.BUSINESS_UNIT,
        )
        target = GraphEntity(
            entity_id=TARGET_ID,
            name="Identity Access System",
            normalized_name="identity access system",
            entity_type=EntityType.SYSTEM,
        )
        evidence = RelationshipEvidence(
            evidence_id=extraction.relationships[0].relationship_id,
            evidence=extraction.relationships[0].evidence,
            provenance=extraction.source,
        )
        return EntityNeighborhood(
            entity=source,
            connections=[
                GraphConnection(
                    source_entity=source,
                    target_entity=target,
                    relationship_type="MANAGES",
                    evidence=[evidence],
                )
            ],
        )


@pytest.fixture
def graph_client() -> Iterator[TestClient]:
    app.dependency_overrides[get_graph_construction_service] = (
        lambda: StubConstructionService()
    )
    app.dependency_overrides[get_knowledge_graph_store] = (
        lambda: StubKnowledgeGraphStore()
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_graph_persistence_api_accepts_typed_extraction(
    graph_client: TestClient,
) -> None:
    response = graph_client.post(
        "/graph/extractions",
        json=_extraction().model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc_1",
        "chunk_id": "chunk_1",
        "entity_count": 2,
        "mention_count": 2,
        "relationship_count": 1,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"entities": [], "relationships": []},
        {"entities": "not-a-list", "relationships": [], "source": {}},
    ],
)
def test_graph_persistence_api_validates_payload(
    graph_client: TestClient, payload: dict[str, object]
) -> None:
    response = graph_client.post("/graph/extractions", json=payload)

    assert response.status_code == 422


def test_graph_neighborhood_api_returns_evidence_and_provenance(
    graph_client: TestClient,
) -> None:
    response = graph_client.get(f"/graph/entities/{ENTITY_ID}/neighbors")

    assert response.status_code == 200
    payload = response.json()
    assert payload["entity"]["entity_id"] == ENTITY_ID
    assert payload["connections"][0]["target_entity"]["entity_id"] == TARGET_ID
    assert payload["connections"][0]["relationship_type"] == "MANAGES"
    assert payload["connections"][0]["evidence"][0]["provenance"]["chunk_id"] == (
        "chunk_1"
    )


def test_graph_neighborhood_api_rejects_invalid_entity_id(
    graph_client: TestClient,
) -> None:
    response = graph_client.get("/graph/entities/not-safe/neighbors")

    assert response.status_code == 422


def test_graph_neighborhood_api_maps_missing_entity(
    graph_client: TestClient,
) -> None:
    class MissingStore:
        def get_entity_neighborhood(self, entity_id: str) -> EntityNeighborhood:
            raise EntityNotFoundError("Entity not found")

    app.dependency_overrides[get_knowledge_graph_store] = lambda: MissingStore()

    response = graph_client.get(f"/graph/entities/{ENTITY_ID}/neighbors")

    assert response.status_code == 404


def test_graph_persistence_api_maps_missing_chunk(
    graph_client: TestClient,
) -> None:
    class MissingChunkService:
        def persist_result(self, result: object) -> GraphPersistenceResponse:
            raise MissingChunkError("Chunk is not stored")

    app.dependency_overrides[get_graph_construction_service] = (
        lambda: MissingChunkService()
    )

    response = graph_client.post(
        "/graph/extractions",
        json=_extraction().model_dump(mode="json"),
    )

    assert response.status_code == 409
