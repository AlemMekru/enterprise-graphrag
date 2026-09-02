"""Deterministic fusion of vector seeds and bounded graph context."""

from __future__ import annotations

from dataclasses import dataclass

from app.graph.knowledge_store import Neo4jKnowledgeGraphStore
from app.models.hybrid import (
    GraphExpansionResult,
    GraphSupportingChunk,
    HybridContextChunk,
    HybridRetrievalResult,
    RetrievalScores,
    RetrievalSource,
)
from app.models.vector import VectorRetrievalResult
from app.retrieval.exceptions import (
    HybridRetrievalConfigurationError,
    RetrievalError,
)
from app.retrieval.vector import VectorRetriever


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Validated ranking weights and graph expansion limits."""

    default_top_k: int = 5
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    max_hops: int = 2
    max_entities: int = 50
    max_relationships: int = 100
    max_supporting_chunks: int = 20

    def __post_init__(self) -> None:
        if not 1 <= self.default_top_k <= 100:
            raise HybridRetrievalConfigurationError(
                "default_top_k must be between 1 and 100"
            )
        if self.vector_weight < 0 or self.graph_weight < 0:
            raise HybridRetrievalConfigurationError(
                "hybrid retrieval weights cannot be negative"
            )
        if self.vector_weight + self.graph_weight <= 0:
            raise HybridRetrievalConfigurationError(
                "at least one hybrid retrieval weight must be positive"
            )
        if self.max_hops not in (1, 2):
            raise HybridRetrievalConfigurationError("max_hops must be 1 or 2")
        if min(
            self.max_entities,
            self.max_relationships,
            self.max_supporting_chunks,
        ) <= 0:
            raise HybridRetrievalConfigurationError(
                "graph expansion bounds must be positive"
            )

    @property
    def normalized_weights(self) -> tuple[float, float]:
        """Return weights normalized to sum to one."""
        total = self.vector_weight + self.graph_weight
        return self.vector_weight / total, self.graph_weight / total


class HybridGraphRetriever:
    """Orchestrate vector retrieval, graph expansion, and score fusion."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_store: Neo4jKnowledgeGraphStore,
        config: HybridRetrievalConfig,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.graph_store = graph_store
        self.config = config

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        graph_hops: int = 1,
    ) -> HybridRetrievalResult:
        """Return citation-ready context without generating an answer."""
        normalized_query = query.strip()
        if not normalized_query:
            raise RetrievalError("query must not be empty")
        result_limit = self.config.default_top_k if top_k is None else top_k
        if not 1 <= result_limit <= 100:
            raise RetrievalError("top_k must be between 1 and 100")
        if graph_hops not in (1, 2) or graph_hops > self.config.max_hops:
            raise RetrievalError(
                f"graph_hops must be between 1 and {self.config.max_hops}"
            )

        vector_results = self.vector_retriever.retrieve(
            normalized_query, result_limit
        )
        graph_result = self.graph_store.expand_seed_chunks(
            [item.chunk_id for item in vector_results],
            hops=graph_hops,
            max_entities=self.config.max_entities,
            max_relationships=self.config.max_relationships,
            max_supporting_chunks=self.config.max_supporting_chunks,
        )
        context = self._fuse(vector_results, graph_result)
        return HybridRetrievalResult(
            query=normalized_query,
            vector_seed_results=vector_results,
            entities=graph_result.entities,
            relationships=graph_result.relationships,
            context=context,
            graph_evidence_found=bool(
                graph_result.entities
                or graph_result.relationships
                or graph_result.supporting_chunks
            ),
        )

    def _fuse(
        self,
        vector_results: list[VectorRetrievalResult],
        graph_result: GraphExpansionResult,
    ) -> list[HybridContextChunk]:
        vector_weight, graph_weight = self.config.normalized_weights
        vector_by_chunk = {
            result.chunk_id: result
            for result in sorted(
                vector_results, key=lambda item: (-item.score, item.chunk_id)
            )
        }
        graph_by_chunk = self._deduplicate_graph_chunks(
            graph_result.supporting_chunks
        )
        chunk_ids = set(vector_by_chunk) | set(graph_by_chunk)
        context: list[HybridContextChunk] = []
        for chunk_id in chunk_ids:
            vector_item = vector_by_chunk.get(chunk_id)
            graph_item = graph_by_chunk.get(chunk_id)
            vector_score = (
                _normalize_similarity(vector_item.score)
                if vector_item is not None
                else None
            )
            graph_score = _graph_score(graph_item) if graph_item else 0.0
            final_score = (
                vector_weight * (vector_score or 0.0)
                + graph_weight * graph_score
            )
            sources: list[RetrievalSource] = []
            if vector_item is not None:
                sources.append(RetrievalSource.VECTOR)
            if graph_item is not None:
                sources.extend(graph_item.retrieval_sources)
            sources = _ordered_unique_sources(sources)

            context.append(
                HybridContextChunk(
                    chunk_id=chunk_id,
                    document_id=(
                        vector_item.document_id
                        if vector_item is not None
                        else graph_item.document_id
                    ),
                    text=(
                        vector_item.text if vector_item is not None else graph_item.text
                    ),
                    chunk_index=(
                        vector_item.chunk_index
                        if vector_item is not None
                        else graph_item.chunk_index
                    ),
                    source=(
                        vector_item.source
                        if vector_item is not None and vector_item.source is not None
                        else graph_item.source if graph_item is not None else None
                    ),
                    source_metadata=(
                        vector_item.source_metadata
                        if vector_item is not None
                        else graph_item.source_metadata
                    ),
                    retrieval_sources=sources,
                    scores=RetrievalScores(
                        vector_score=vector_score,
                        graph_score=round(graph_score, 6),
                        final_score=round(final_score, 6),
                    ),
                    graph_distance=(
                        graph_item.graph_distance if graph_item is not None else None
                    ),
                    related_entity_ids=(
                        graph_item.entity_ids if graph_item is not None else []
                    ),
                    relationship_ids=(
                        graph_item.relationship_ids if graph_item is not None else []
                    ),
                )
            )
        return sorted(
            context,
            key=lambda item: (-item.scores.final_score, item.chunk_id),
        )

    @staticmethod
    def _deduplicate_graph_chunks(
        chunks: list[GraphSupportingChunk],
    ) -> dict[str, GraphSupportingChunk]:
        merged: dict[str, GraphSupportingChunk] = {}
        for chunk in chunks:
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk.model_copy(deep=True)
                continue
            existing.graph_distance = min(
                existing.graph_distance, chunk.graph_distance
            )
            existing.entity_ids = sorted(
                set(existing.entity_ids) | set(chunk.entity_ids)
            )
            existing.relationship_ids = sorted(
                set(existing.relationship_ids) | set(chunk.relationship_ids)
            )
            existing.retrieval_sources = _ordered_unique_sources(
                existing.retrieval_sources + chunk.retrieval_sources
            )
        return merged


def _normalize_similarity(value: float) -> float:
    """Clamp Neo4j's normalized similarity score to the fusion range."""
    return round(max(0.0, min(1.0, value)), 6)


def _graph_score(chunk: GraphSupportingChunk) -> float:
    """Score nearer graph support higher, with bounded independent-support boosts."""
    distance_signal = 1.0 / (chunk.graph_distance + 1)
    entity_support = min(0.15, 0.05 * max(0, len(chunk.entity_ids) - 1))
    relationship_support = min(0.2, 0.1 * len(chunk.relationship_ids))
    return min(1.0, distance_signal + entity_support + relationship_support)


def _ordered_unique_sources(
    sources: list[RetrievalSource],
) -> list[RetrievalSource]:
    order = {source: index for index, source in enumerate(RetrievalSource)}
    return sorted(set(sources), key=order.__getitem__)
