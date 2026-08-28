"""Tests for Phase 2 environment configuration."""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.graph.exceptions import VectorIndexConfigurationError
from app.graph.vector_store import VectorIndexConfig


def test_phase_two_configuration_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "openai"
    assert settings.embedding_dimension == 1_536
    assert settings.neo4j_vector_index_name == "chunk_embedding_index"
    assert settings.vector_similarity_function == "cosine"
    assert settings.retrieval_top_k == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_dimension": 0},
        {"embedding_dimension": 4_097},
        {"retrieval_top_k": 0},
        {"retrieval_top_k": 101},
        {"neo4j_vector_index_name": "invalid-index-name"},
        {"neo4j_vector_index_name": "123_index"},
    ],
)
def test_invalid_phase_two_configuration(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(**overrides, _env_file=None)


@pytest.mark.parametrize(
    "name,dimension",
    [("invalid-name", 3), ("chunk_index", 0), ("chunk_index", 4_097)],
)
def test_vector_index_config_rejects_unsafe_values(
    name: str, dimension: int
) -> None:
    with pytest.raises(VectorIndexConfigurationError):
        VectorIndexConfig(name=name, dimension=dimension)
