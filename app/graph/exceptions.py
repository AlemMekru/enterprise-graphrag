"""Neo4j persistence and vector-index exceptions."""


class VectorStoreError(Exception):
    """Base exception for Neo4j vector-store failures."""


class VectorIndexConfigurationError(VectorStoreError):
    """Raised when an existing vector index conflicts with configuration."""
