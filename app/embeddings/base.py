"""Embedding provider contract."""

from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Contract implemented by embedding backends."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed input texts in the same order they were provided."""
        ...
