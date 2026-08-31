"""Orchestration for persisting validated Phase 3 results."""

from collections.abc import Iterable

from app.graph.knowledge_store import Neo4jKnowledgeGraphStore
from app.models.graph import ExtractionResult, GraphPersistenceResponse


class GraphConstructionService:
    """Initialize schema and persist extraction results without re-extraction."""

    def __init__(self, store: Neo4jKnowledgeGraphStore) -> None:
        self.store = store

    def persist_result(self, result: ExtractionResult) -> GraphPersistenceResponse:
        """Persist one extraction result after idempotent schema setup."""
        self.store.ensure_schema()
        return self.store.persist_extraction_result(result)

    def persist_results(
        self, results: Iterable[ExtractionResult]
    ) -> list[GraphPersistenceResponse]:
        """Persist multiple independently transactional extraction results."""
        self.store.ensure_schema()
        return [self.store.persist_extraction_result(result) for result in results]
