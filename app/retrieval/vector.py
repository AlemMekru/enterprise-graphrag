"""Semantic retrieval over Neo4j chunk vectors."""

from app.embeddings.base import EmbeddingProvider
from app.embeddings.exceptions import EmbeddingDimensionError, EmbeddingProviderError
from app.graph.vector_store import Neo4jVectorStore
from app.models.vector import VectorRetrievalResult
from app.retrieval.exceptions import RetrievalError


class VectorRetriever:
    """Embed a query and retrieve the most similar stored chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: Neo4jVectorStore,
        embedding_dimension: int,
        default_top_k: int,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.embedding_dimension = embedding_dimension
        self.default_top_k = default_top_k

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[VectorRetrievalResult]:
        """Run query embedding followed by Neo4j vector search."""
        normalized_query = query.strip()
        if not normalized_query:
            raise RetrievalError("query must not be empty")
        result_limit = self.default_top_k if top_k is None else top_k
        if result_limit <= 0:
            raise RetrievalError("top_k must be greater than zero")

        vectors = self.embedding_provider.embed([normalized_query])
        if len(vectors) != 1:
            raise EmbeddingProviderError(
                "Embedding provider did not return exactly one query vector"
            )
        embedding = vectors[0]
        if len(embedding) != self.embedding_dimension:
            raise EmbeddingDimensionError(
                f"Expected {self.embedding_dimension} query dimensions, "
                f"received {len(embedding)}"
            )
        return self.vector_store.vector_search(embedding, result_limit)
