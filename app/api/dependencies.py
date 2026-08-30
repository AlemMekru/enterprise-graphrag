"""FastAPI dependency wiring for external services."""

from collections.abc import Iterator

from fastapi import Depends, HTTPException, status
from neo4j import GraphDatabase

from app.config import Settings, get_settings
from app.embeddings.exceptions import EmbeddingConfigurationError
from app.embeddings.factory import create_embedding_provider
from app.extraction.exceptions import ExtractionConfigurationError
from app.extraction.factory import create_graph_extraction_provider
from app.extraction.service import GraphExtractionService
from app.graph.vector_store import Neo4jVectorStore, VectorIndexConfig
from app.retrieval.vector import VectorRetriever


def get_graph_extraction_service(
    settings: Settings = Depends(get_settings),
) -> GraphExtractionService:
    """Build the configured structured graph extraction service."""
    try:
        provider = create_graph_extraction_provider(settings)
    except ExtractionConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return GraphExtractionService(provider)


def get_vector_retriever(
    settings: Settings = Depends(get_settings),
) -> Iterator[VectorRetriever]:
    """Build request-scoped vector retrieval dependencies."""
    if settings.neo4j_password is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NEO4J_PASSWORD is required for vector retrieval",
        )
    try:
        provider = create_embedding_provider(settings)
    except EmbeddingConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    store = Neo4jVectorStore(
        driver=driver,
        database=settings.neo4j_database,
        index_config=VectorIndexConfig(
            name=settings.neo4j_vector_index_name,
            dimension=settings.embedding_dimension,
            similarity_function=settings.vector_similarity_function,
        ),
    )
    try:
        yield VectorRetriever(
            embedding_provider=provider,
            vector_store=store,
            embedding_dimension=settings.embedding_dimension,
            default_top_k=settings.retrieval_top_k,
        )
    finally:
        driver.close()
