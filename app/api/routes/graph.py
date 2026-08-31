"""Knowledge-graph persistence and neighborhood endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.dependencies import (
    get_graph_construction_service,
    get_knowledge_graph_store,
)
from app.graph.construction import GraphConstructionService
from app.graph.exceptions import (
    EntityNotFoundError,
    KnowledgeGraphInputError,
    KnowledgeGraphStoreError,
    MissingChunkError,
)
from app.graph.knowledge_store import Neo4jKnowledgeGraphStore
from app.models.graph import (
    EntityNeighborhood,
    ExtractionResult,
    GraphPersistenceResponse,
)

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


@router.post("/extractions", response_model=GraphPersistenceResponse)
def persist_extraction(
    extraction: ExtractionResult,
    service: GraphConstructionService = Depends(get_graph_construction_service),
) -> GraphPersistenceResponse:
    """Persist one already validated extraction without accepting Cypher."""
    try:
        return service.persist_result(extraction)
    except MissingChunkError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except KnowledgeGraphInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except KnowledgeGraphStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge-graph persistence is temporarily unavailable",
        ) from exc


@router.get(
    "/entities/{entity_id}/neighbors",
    response_model=EntityNeighborhood,
)
def get_entity_neighbors(
    entity_id: str = Path(pattern=r"^entity_[a-f0-9]{64}$"),
    store: Neo4jKnowledgeGraphStore = Depends(get_knowledge_graph_store),
) -> EntityNeighborhood:
    """Return directly connected semantic entities and supporting evidence."""
    try:
        return store.get_entity_neighborhood(entity_id)
    except EntityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except KnowledgeGraphStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge-graph query is temporarily unavailable",
        ) from exc
