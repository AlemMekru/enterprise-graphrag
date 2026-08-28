"""Map Phase 1 document chunks to embedding records."""

from collections.abc import Sequence

from app.embeddings.base import EmbeddingProvider
from app.embeddings.exceptions import EmbeddingDimensionError, EmbeddingProviderError
from app.models.document import DocumentChunk
from app.models.vector import EmbeddedChunk


class ChunkEmbeddingService:
    """Generate validated embeddings while preserving chunk provenance."""

    def __init__(self, provider: EmbeddingProvider, dimension: int) -> None:
        self.provider = provider
        self.dimension = dimension

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> list[EmbeddedChunk]:
        """Embed a chunk batch and retain every Phase 1 field."""
        vectors = self.provider.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise EmbeddingProviderError(
                "Embedding provider returned a different number of vectors than inputs"
            )

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.dimension:
                raise EmbeddingDimensionError(
                    f"Expected {self.dimension} dimensions, received {len(vector)} "
                    f"for chunk {chunk.chunk_id}"
                )
            embedded_chunks.append(
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
            )
        return embedded_chunks
