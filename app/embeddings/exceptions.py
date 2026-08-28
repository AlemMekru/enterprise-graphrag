"""Embedding-domain exceptions."""


class EmbeddingError(Exception):
    """Base exception for embedding failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when provider configuration is missing or incompatible."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an external embedding provider request fails."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when returned vectors do not match the configured dimension."""
