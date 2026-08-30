"""Tests for OpenAI-compatible structured extraction."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError

from app.extraction.exceptions import (
    ExtractionProviderError,
    MalformedExtractionError,
)
from app.extraction.openai_provider import OpenAIGraphExtractionProvider
from app.extraction.prompt import GRAPH_EXTRACTION_SYSTEM_PROMPT
from app.models.graph import ProviderEntity, ProviderExtraction


def _client_with_message(message: object) -> MagicMock:
    client = MagicMock()
    client.chat.completions.parse.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )
    return client


def test_openai_provider_uses_pydantic_structured_output() -> None:
    parsed = ProviderExtraction(
        entities=[ProviderEntity(name="Northstar", entity_type="ORGANIZATION")]
    )
    client = _client_with_message(SimpleNamespace(parsed=parsed, refusal=None))
    provider = OpenAIGraphExtractionProvider(client, model="gpt-test")

    result = provider.extract("Northstar operates a service.")

    assert result == parsed
    call = client.chat.completions.parse.call_args
    assert call.kwargs["model"] == "gpt-test"
    assert call.kwargs["response_format"] is ProviderExtraction
    assert call.kwargs["temperature"] == 0
    assert call.kwargs["messages"][0]["content"] == GRAPH_EXTRACTION_SYSTEM_PROMPT
    assert call.kwargs["messages"][1]["content"] == (
        "Northstar operates a service."
    )


def test_openai_provider_rejects_missing_parsed_response() -> None:
    client = _client_with_message(SimpleNamespace(parsed=None, refusal=None))

    with pytest.raises(MalformedExtractionError, match="no structured result"):
        OpenAIGraphExtractionProvider(client, "gpt-test").extract("text")


def test_openai_provider_handles_refusal() -> None:
    client = _client_with_message(
        SimpleNamespace(parsed=None, refusal="Unable to process")
    )

    with pytest.raises(ExtractionProviderError, match="refused"):
        OpenAIGraphExtractionProvider(client, "gpt-test").extract("text")


def test_openai_provider_wraps_api_failures() -> None:
    client = MagicMock()
    client.chat.completions.parse.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )

    with pytest.raises(ExtractionProviderError, match="provider failed"):
        OpenAIGraphExtractionProvider(client, "gpt-test").extract("text")
