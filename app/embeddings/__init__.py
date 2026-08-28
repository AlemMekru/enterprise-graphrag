"""Provider-neutral embedding generation."""

from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import create_embedding_provider
from app.embeddings.service import ChunkEmbeddingService

__all__ = [
    "ChunkEmbeddingService",
    "EmbeddingProvider",
    "create_embedding_provider",
]
