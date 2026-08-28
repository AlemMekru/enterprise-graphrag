"""OpenAI-compatible embedding provider."""

from typing import Protocol, Sequence

from openai import OpenAIError

from app.embeddings.exceptions import EmbeddingProviderError


class _EmbeddingDatum(Protocol):
    index: int
    embedding: Sequence[float]


class _EmbeddingResponse(Protocol):
    data: Sequence[_EmbeddingDatum]


class _EmbeddingResource(Protocol):
    def create(self, **kwargs: object) -> _EmbeddingResponse:
        """Create embeddings using an OpenAI-compatible client."""
        ...


class _OpenAICompatibleClient(Protocol):
    embeddings: _EmbeddingResource


class OpenAIEmbeddingProvider:
    """Generate embeddings through OpenAI or Azure OpenAI clients."""

    def __init__(
        self,
        client: _OpenAICompatibleClient,
        model: str,
        dimension: int,
    ) -> None:
        self.client = client
        self.model = model
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate ordered embeddings for a batch of texts."""
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=list(texts),
                model=self.model,
                dimensions=self.dimension,
            )
            data = sorted(response.data, key=lambda item: item.index)
            return [list(item.embedding) for item in data]
        except OpenAIError as exc:
            raise EmbeddingProviderError("Embedding provider request failed") from exc
