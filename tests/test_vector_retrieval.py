"""Unit tests for semantic vector retrieval."""

from unittest.mock import MagicMock

import pytest

from app.embeddings.exceptions import EmbeddingDimensionError
from app.models.vector import VectorRetrievalResult
from app.retrieval.vector import VectorRetriever


class QueryEmbeddingProvider:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding
        self.inputs: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(list(texts))
        return [self.embedding]


def _result() -> VectorRetrievalResult:
    return VectorRetrievalResult(
        chunk_id="chunk_1",
        document_id="doc_1",
        text="Retention is seven years.",
        score=0.94,
        source_metadata={"filename": "policy.md"},
    )


def test_retriever_embeds_query_and_uses_requested_top_k() -> None:
    provider = QueryEmbeddingProvider([0.1, 0.2, 0.3])
    store = MagicMock()
    store.vector_search.return_value = [_result()]
    retriever = VectorRetriever(provider, store, embedding_dimension=3, default_top_k=5)

    results = retriever.retrieve("  retention policy  ", top_k=2)

    assert results == [_result()]
    assert provider.inputs == [["retention policy"]]
    store.vector_search.assert_called_once_with([0.1, 0.2, 0.3], 2)


def test_retriever_uses_configured_default_top_k() -> None:
    store = MagicMock()
    store.vector_search.return_value = []
    retriever = VectorRetriever(
        QueryEmbeddingProvider([0.1, 0.2, 0.3]),
        store,
        embedding_dimension=3,
        default_top_k=7,
    )

    retriever.retrieve("policy")

    assert store.vector_search.call_args.args[1] == 7


def test_retriever_rejects_wrong_query_embedding_dimension() -> None:
    retriever = VectorRetriever(
        QueryEmbeddingProvider([0.1]),
        MagicMock(),
        embedding_dimension=3,
        default_top_k=5,
    )

    with pytest.raises(EmbeddingDimensionError):
        retriever.retrieve("policy")
