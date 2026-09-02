"""Unit tests for deterministic hybrid GraphRAG orchestration and fusion."""

from unittest.mock import MagicMock

import pytest

from app.embeddings.exceptions import EmbeddingProviderError
from app.graph.exceptions import KnowledgeGraphStoreError
from app.models.graph import GraphEntity
from app.models.hybrid import (
    GraphExpandedEntity,
    GraphExpandedRelationship,
    GraphExpansionResult,
    GraphSupportingChunk,
    RetrievalSource,
)
from app.models.vector import VectorRetrievalResult
from app.retrieval.exceptions import (
    HybridRetrievalConfigurationError,
    RetrievalError,
)
from app.retrieval.hybrid import HybridGraphRetriever, HybridRetrievalConfig


def _vector(
    chunk_id: str = "chunk_seed", score: float = 0.8
) -> VectorRetrievalResult:
    return VectorRetrievalResult(
        chunk_id=chunk_id,
        document_id="doc_1",
        text=f"Text for {chunk_id}",
        score=score,
        chunk_index=1,
        source="/data/policy.md",
        source_metadata={"filename": "policy.md"},
    )


def _support(
    chunk_id: str = "chunk_support",
    distance: int = 1,
    entity_ids: list[str] | None = None,
    relationship_ids: list[str] | None = None,
) -> GraphSupportingChunk:
    relationships = relationship_ids or []
    sources = [RetrievalSource.GRAPH_ENTITY]
    if relationships:
        sources += [
            RetrievalSource.GRAPH_RELATIONSHIP,
            RetrievalSource.RELATIONSHIP_EVIDENCE,
        ]
    return GraphSupportingChunk(
        chunk_id=chunk_id,
        document_id="doc_2" if chunk_id != "chunk_seed" else "doc_1",
        text=f"Text for {chunk_id}",
        chunk_index=3,
        source="/data/standard.md",
        source_metadata={"filename": "standard.md", "page": 2},
        graph_distance=distance,
        entity_ids=entity_ids or ["entity_a"],
        relationship_ids=relationships,
        retrieval_sources=sources,
    )


def _retriever(
    vectors: list[VectorRetrievalResult] | None = None,
    graph: GraphExpansionResult | None = None,
    config: HybridRetrievalConfig | None = None,
) -> tuple[HybridGraphRetriever, MagicMock, MagicMock]:
    vector_retriever = MagicMock()
    vector_retriever.retrieve.return_value = vectors if vectors is not None else [_vector()]
    graph_store = MagicMock()
    graph_store.expand_seed_chunks.return_value = graph or GraphExpansionResult()
    return (
        HybridGraphRetriever(
            vector_retriever,
            graph_store,
            config or HybridRetrievalConfig(),
        ),
        vector_retriever,
        graph_store,
    )


def test_reuses_vector_retriever_and_passes_seed_ids_to_graph_store() -> None:
    retriever, vector, graph = _retriever(vectors=[_vector("a"), _vector("b")])

    retriever.retrieve("  governed systems  ", top_k=2, graph_hops=2)

    vector.retrieve.assert_called_once_with("governed systems", 2)
    graph.expand_seed_chunks.assert_called_once_with(
        ["a", "b"],
        hops=2,
        max_entities=50,
        max_relationships=100,
        max_supporting_chunks=20,
    )


def test_graph_expansion_adds_non_vector_supporting_chunk() -> None:
    retriever, _, _ = _retriever(
        graph=GraphExpansionResult(supporting_chunks=[_support()])
    )

    result = retriever.retrieve("systems")

    assert {item.chunk_id for item in result.context} == {
        "chunk_seed",
        "chunk_support",
    }
    graph_item = next(item for item in result.context if item.chunk_id == "chunk_support")
    assert graph_item.scores.vector_score is None
    assert graph_item.scores.graph_score == 0.5


def test_duplicate_chunk_signals_are_merged_once_with_explanations() -> None:
    duplicate_one = _support(
        "chunk_seed", entity_ids=["entity_a"], relationship_ids=["rel_1"]
    )
    duplicate_two = _support(
        "chunk_seed", distance=0, entity_ids=["entity_b"], relationship_ids=["rel_2"]
    )
    retriever, _, _ = _retriever(
        graph=GraphExpansionResult(
            supporting_chunks=[duplicate_one, duplicate_two]
        )
    )

    result = retriever.retrieve("systems")

    assert len(result.context) == 1
    item = result.context[0]
    assert item.related_entity_ids == ["entity_a", "entity_b"]
    assert item.relationship_ids == ["rel_1", "rel_2"]
    assert item.retrieval_sources == list(RetrievalSource)
    assert item.graph_distance == 0


def test_fusion_uses_normalized_configured_weights() -> None:
    config = HybridRetrievalConfig(vector_weight=3, graph_weight=1)
    retriever, _, _ = _retriever(
        vectors=[_vector(score=0.8)],
        graph=GraphExpansionResult(
            supporting_chunks=[_support("chunk_seed", distance=1)]
        ),
        config=config,
    )

    item = retriever.retrieve("systems").context[0]

    assert item.scores.vector_score == 0.8
    assert item.scores.graph_score == 0.5
    assert item.scores.final_score == 0.725


def test_relationship_and_independent_entity_support_raise_graph_score() -> None:
    retriever, _, _ = _retriever(
        graph=GraphExpansionResult(
            supporting_chunks=[
                _support(
                    entity_ids=["a", "b", "c"],
                    relationship_ids=["r1", "r2"],
                )
            ]
        )
    )

    graph_item = next(
        item
        for item in retriever.retrieve("systems").context
        if item.chunk_id == "chunk_support"
    )
    assert graph_item.scores.graph_score == 0.8


def test_ranking_is_deterministic_with_chunk_id_tie_breaker() -> None:
    retriever, _, _ = _retriever(
        vectors=[_vector("chunk_b"), _vector("chunk_a")]
    )

    first = retriever.retrieve("systems").context
    second = retriever.retrieve("systems").context

    assert [item.chunk_id for item in first] == ["chunk_a", "chunk_b"]
    assert first == second


def test_vector_similarity_is_clamped_before_fusion() -> None:
    retriever, _, _ = _retriever(vectors=[_vector(score=1.4)])

    item = retriever.retrieve("systems").context[0]

    assert item.scores.vector_score == 1.0
    assert item.scores.final_score == 0.7


def test_vector_only_fallback_is_valid_and_distinct_from_graph_failure() -> None:
    retriever, _, _ = _retriever()

    result = retriever.retrieve("systems")

    assert result.graph_evidence_found is False
    assert result.context[0].retrieval_sources == [RetrievalSource.VECTOR]
    assert result.context[0].scores.graph_score == 0


def test_entities_and_relationships_remain_structured_in_result() -> None:
    entity = GraphExpandedEntity(
        entity=GraphEntity(
            entity_id="entity_a",
            name="Security Policy",
            normalized_name="security policy",
            entity_type="POLICY",
        ),
        graph_distance=0,
        seed_chunk_ids=["chunk_seed"],
    )
    relationship = GraphExpandedRelationship(
        relationship_id="semantic_1",
        source_entity_id="entity_a",
        target_entity_id="entity_b",
        relationship_type="GOVERNS",
        graph_distance=1,
    )
    retriever, _, _ = _retriever(
        graph=GraphExpansionResult(
            entities=[entity], relationships=[relationship]
        )
    )

    result = retriever.retrieve("systems")

    assert result.entities == [entity]
    assert result.relationships == [relationship]
    assert result.graph_evidence_found is True


def test_provenance_is_preserved_for_vector_and_graph_chunks() -> None:
    retriever, _, _ = _retriever(
        graph=GraphExpansionResult(supporting_chunks=[_support()])
    )

    result = retriever.retrieve("systems")
    seed = next(item for item in result.context if item.chunk_id == "chunk_seed")
    support = next(item for item in result.context if item.chunk_id == "chunk_support")

    assert (seed.document_id, seed.chunk_index, seed.source) == (
        "doc_1",
        1,
        "/data/policy.md",
    )
    assert support.source_metadata == {"filename": "standard.md", "page": 2}


@pytest.mark.parametrize("hops", [0, 3])
def test_rejects_unsafe_hop_input_before_external_calls(hops: int) -> None:
    retriever, vector, graph = _retriever()

    with pytest.raises(RetrievalError, match="graph_hops"):
        retriever.retrieve("systems", graph_hops=hops)

    vector.retrieve.assert_not_called()
    graph.expand_seed_chunks.assert_not_called()


def test_configured_maximum_hops_is_enforced() -> None:
    retriever, _, _ = _retriever(config=HybridRetrievalConfig(max_hops=1))

    with pytest.raises(RetrievalError, match="between 1 and 1"):
        retriever.retrieve("systems", graph_hops=2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"default_top_k": 0},
        {"vector_weight": -0.1},
        {"vector_weight": 0, "graph_weight": 0},
        {"max_hops": 3},
        {"max_entities": 0},
        {"max_relationships": 0},
        {"max_supporting_chunks": 0},
    ],
)
def test_invalid_hybrid_configuration_is_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(HybridRetrievalConfigurationError):
        HybridRetrievalConfig(**overrides)


def test_embedding_failure_is_not_hidden_or_followed_by_graph_query() -> None:
    retriever, vector, graph = _retriever()
    vector.retrieve.side_effect = EmbeddingProviderError("provider unavailable")

    with pytest.raises(EmbeddingProviderError):
        retriever.retrieve("systems")

    graph.expand_seed_chunks.assert_not_called()


def test_neo4j_graph_failure_is_distinct_from_empty_graph_result() -> None:
    retriever, _, graph = _retriever()
    graph.expand_seed_chunks.side_effect = KnowledgeGraphStoreError(
        "database unavailable"
    )

    with pytest.raises(KnowledgeGraphStoreError, match="database unavailable"):
        retriever.retrieve("systems")
