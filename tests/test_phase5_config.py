"""Configuration validation for bounded hybrid retrieval."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_phase_five_configuration_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.hybrid_default_top_k == 5
    assert settings.hybrid_vector_weight == 0.7
    assert settings.hybrid_graph_weight == 0.3
    assert settings.graph_max_hops == 2
    assert settings.graph_max_entities == 50
    assert settings.graph_max_relationships == 100
    assert settings.graph_max_supporting_chunks == 20


@pytest.mark.parametrize(
    "overrides",
    [
        {"hybrid_default_top_k": 0},
        {"hybrid_default_top_k": 101},
        {"hybrid_vector_weight": -0.1},
        {"hybrid_vector_weight": 0, "hybrid_graph_weight": 0},
        {"graph_max_hops": 0},
        {"graph_max_hops": 3},
        {"graph_max_entities": 0},
        {"graph_max_entities": 501},
        {"graph_max_relationships": 0},
        {"graph_max_relationships": 1_001},
        {"graph_max_supporting_chunks": 0},
        {"graph_max_supporting_chunks": 201},
    ],
)
def test_invalid_phase_five_configuration_is_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Settings(**overrides, _env_file=None)
