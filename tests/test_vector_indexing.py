"""Unit tests for vector indexing orchestration."""

from unittest.mock import MagicMock

from app.indexing.pipeline import VectorIndexingPipeline
from app.models.document import Document, DocumentChunk
from app.models.vector import EmbeddedChunk


def test_indexing_pipeline_embeds_and_persists_phase_one_chunks() -> None:
    document = Document(
        document_id="doc_1",
        source="/data/policy.md",
        content="Retention policy",
    )
    chunk = DocumentChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        text="Retention policy",
        chunk_index=0,
    )
    embedded = EmbeddedChunk(**chunk.model_dump(), embedding=[0.1, 0.2])
    embedding_service = MagicMock()
    embedding_service.embed_chunks.return_value = [embedded]
    vector_store = MagicMock()
    pipeline = VectorIndexingPipeline(embedding_service, vector_store)

    result = pipeline.index(document, [chunk])

    assert result == [embedded]
    vector_store.ensure_schema.assert_called_once_with()
    embedding_service.embed_chunks.assert_called_once_with([chunk])
    vector_store.upsert_document_chunks.assert_called_once_with(document, [embedded])
