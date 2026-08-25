"""Domain-specific ingestion errors."""


class IngestionError(Exception):
    """Base exception for expected document-ingestion failures."""


class UnsupportedDocumentTypeError(IngestionError):
    """Raised when a document extension is not supported."""


class EmptyDocumentError(IngestionError):
    """Raised when a document contains no extractable text."""


class DocumentReadError(IngestionError):
    """Raised when a document cannot be opened or parsed."""


class InvalidChunkConfigurationError(IngestionError):
    """Raised when chunk size or overlap settings are invalid."""
