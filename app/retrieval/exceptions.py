"""Retrieval-domain exceptions."""


class RetrievalError(Exception):
    """Raised when semantic retrieval cannot be completed."""


class HybridRetrievalConfigurationError(RetrievalError):
    """Raised when hybrid retrieval bounds or weights are invalid."""
