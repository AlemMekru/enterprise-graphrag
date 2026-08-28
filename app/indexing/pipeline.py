"""Orchestrate chunk embedding and Neo4j vector persistence."""

from collections.abc import Sequence

from app.embeddings.service import ChunkEmbeddingService
from app.graph.vector_store import Neo4jVectorStore
from app.models.document import Document, DocumentChunk
from app.models.vector import EmbeddedChunk


class VectorIndexingPipeline:
    """Embed and idempotently index Phase 1 chunks in Neo4j."""

    def __init__(
        self,
        embedding_service: ChunkEmbeddingService,
        vector_store: Neo4jVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index(
        self, document: Document, chunks: Sequence[DocumentChunk]
    ) -> list[EmbeddedChunk]:
        """Ensure schema, generate embeddings, and upsert chunk records."""
        self.vector_store.ensure_schema()
        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        self.vector_store.upsert_document_chunks(document, embedded_chunks)
        return embedded_chunks
