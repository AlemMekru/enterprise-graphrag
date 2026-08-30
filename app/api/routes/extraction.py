"""Entity and relationship extraction endpoint."""

import hashlib

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_graph_extraction_service
from app.extraction.exceptions import GraphExtractionError
from app.extraction.service import GraphExtractionService
from app.models.document import DocumentChunk
from app.models.graph import ExtractionResult, GraphExtractionRequest

router = APIRouter(prefix="/extract", tags=["Extraction"])


@router.post("/graph", response_model=ExtractionResult)
def extract_graph(
    request: GraphExtractionRequest,
    service: GraphExtractionService = Depends(get_graph_extraction_service),
) -> ExtractionResult:
    """Return validated graph candidates without persisting them."""
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="text must not be blank",
        )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_id = request.document_id or f"api_document_{digest}"
    chunk_id = request.chunk_id or f"api_chunk_{digest}"
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=request.chunk_index,
        source_metadata=request.source_metadata,
    )
    try:
        return service.extract_chunk(chunk)
    except GraphExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Structured graph extraction failed",
        ) from exc
