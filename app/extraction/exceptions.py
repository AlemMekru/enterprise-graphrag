"""Graph extraction exceptions."""


class GraphExtractionError(Exception):
    """Base exception for graph extraction failures."""


class ExtractionConfigurationError(GraphExtractionError):
    """Raised when the configured provider cannot be constructed."""


class ExtractionProviderError(GraphExtractionError):
    """Raised when an external extraction provider fails or refuses."""


class MalformedExtractionError(GraphExtractionError):
    """Raised when a provider response does not match the extraction schema."""


class GraphValidationError(GraphExtractionError):
    """Raised when extracted graph candidates are structurally invalid."""
