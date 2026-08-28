"""Unit tests for embedding providers and chunk mapping."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import Settings
from app.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingProviderError,
)
from app.embeddings.factory import create_embedding_provider
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.embeddings.service import ChunkEmbeddingService
from app.models.document import DocumentChunk


class FakeEmbeddingProvider:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self.vectors


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_0",
            document_id="doc_1",
            text="Retention is seven years.",
            chunk_index=0,
            source_metadata={"filename": "policy.md"},
        ),
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            text="Legal holds suspend deletion.",
            chunk_index=1,
            source_metadata={"filename": "policy.md"},
        ),
    ]


def test_openai_compatible_provider_preserves_response_order() -> None:
    embeddings = SimpleNamespace()
    embeddings.create = lambda **_: SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ]
    )
    client = SimpleNamespace(embeddings=embeddings)
    provider = OpenAIEmbeddingProvider(client, model="embedding-model", dimension=2)

    result = provider.embed(["first", "second"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_chunk_embedding_is_deterministic_with_mocked_provider() -> None:
    provider = FakeEmbeddingProvider([[0.1, 0.2], [0.3, 0.4]])
    service = ChunkEmbeddingService(provider=provider, dimension=2)

    first = service.embed_chunks(_chunks())
    second = service.embed_chunks(_chunks())

    assert first == second
    assert provider.calls == [
        ["Retention is seven years.", "Legal holds suspend deletion."],
        ["Retention is seven years.", "Legal holds suspend deletion."],
    ]
    assert first[0].chunk_id == "chunk_0"
    assert first[0].document_id == "doc_1"
    assert first[0].chunk_index == 0
    assert first[0].source_metadata == {"filename": "policy.md"}


def test_chunk_embedding_rejects_wrong_vector_count() -> None:
    service = ChunkEmbeddingService(FakeEmbeddingProvider([[0.1, 0.2]]), dimension=2)

    with pytest.raises(EmbeddingProviderError):
        service.embed_chunks(_chunks())


def test_chunk_embedding_rejects_wrong_dimension() -> None:
    service = ChunkEmbeddingService(
        FakeEmbeddingProvider([[0.1], [0.2]]), dimension=2
    )

    with pytest.raises(EmbeddingDimensionError):
        service.embed_chunks(_chunks())


def test_openai_provider_requires_api_key() -> None:
    settings = Settings(openai_api_key=None, _env_file=None)

    with pytest.raises(EmbeddingConfigurationError, match="OPENAI_API_KEY"):
        create_embedding_provider(settings)


def test_azure_provider_reports_missing_configuration() -> None:
    settings = Settings(
        embedding_provider="azure_openai",
        azure_openai_api_key=None,
        azure_openai_endpoint=None,
        azure_openai_api_version=None,
        azure_openai_embedding_deployment=None,
        _env_file=None,
    )

    with pytest.raises(EmbeddingConfigurationError, match="AZURE_OPENAI_API_KEY"):
        create_embedding_provider(settings)


def test_azure_provider_uses_configured_deployment() -> None:
    settings = Settings(
        embedding_provider="azure_openai",
        embedding_dimension=3,
        azure_openai_api_key="azure-secret",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_version="2025-04-01-preview",
        azure_openai_embedding_deployment="enterprise-embeddings",
        _env_file=None,
    )

    with patch("app.embeddings.factory.AzureOpenAI") as client_class:
        provider = create_embedding_provider(settings)

    client_class.assert_called_once_with(
        api_key="azure-secret",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2025-04-01-preview",
    )
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model == "enterprise-embeddings"
    assert provider.dimension == 3
