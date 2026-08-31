"""Neo4j persistence and vector-index exceptions."""


class VectorStoreError(Exception):
    """Base exception for Neo4j vector-store failures."""


class VectorIndexConfigurationError(VectorStoreError):
    """Raised when an existing vector index conflicts with configuration."""


class KnowledgeGraphError(Exception):
    """Base exception for knowledge-graph persistence and queries."""


class KnowledgeGraphInputError(KnowledgeGraphError):
    """Raised when graph candidates are unsafe or structurally invalid."""


class MissingChunkError(KnowledgeGraphError):
    """Raised when extraction provenance does not resolve to a stored chunk."""


class MissingEntityEndpointError(KnowledgeGraphInputError):
    """Raised when a semantic relationship endpoint is absent."""


class EntityNotFoundError(KnowledgeGraphError):
    """Raised when a requested entity is not present in the graph."""


class KnowledgeGraphStoreError(KnowledgeGraphError):
    """Raised when Neo4j graph persistence or querying fails."""
