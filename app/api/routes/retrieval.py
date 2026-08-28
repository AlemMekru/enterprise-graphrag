"""Semantic retrieval endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_vector_retriever
from app.embeddings.exceptions import EmbeddingError
from app.graph.exceptions import VectorStoreError
from app.models.vector import VectorRetrievalRequest, VectorRetrievalResponse
from app.retrieval.exceptions import RetrievalError
from app.retrieval.vector import VectorRetriever

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])


@router.post("/vector", response_model=VectorRetrievalResponse)
def retrieve_vector(
    request: VectorRetrievalRequest,
    retriever: VectorRetriever = Depends(get_vector_retriever),
) -> VectorRetrievalResponse:
    """Retrieve semantically similar chunks without LLM generation."""
    try:
        results = retriever.retrieve(request.query, request.top_k)
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (EmbeddingError, VectorStoreError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector retrieval is temporarily unavailable",
        ) from exc
    return VectorRetrievalResponse(query=request.query, results=results)
