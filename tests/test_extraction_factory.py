"""Tests for OpenAI and Azure extraction provider configuration."""

from unittest.mock import patch

import pytest

from app.config import Settings
from app.extraction.exceptions import ExtractionConfigurationError
from app.extraction.factory import create_graph_extraction_provider
from app.extraction.openai_provider import OpenAIGraphExtractionProvider


def test_openai_extraction_provider_requires_api_key() -> None:
    settings = Settings(llm_provider="openai", openai_api_key=None, _env_file=None)

    with pytest.raises(ExtractionConfigurationError, match="OPENAI_API_KEY"):
        create_graph_extraction_provider(settings)


def test_openai_extraction_provider_uses_chat_model() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-secret",
        openai_chat_model="gpt-extraction",
        _env_file=None,
    )

    with patch("app.extraction.factory.OpenAI") as client_class:
        provider = create_graph_extraction_provider(settings)

    client_class.assert_called_once_with(api_key="test-secret")
    assert isinstance(provider, OpenAIGraphExtractionProvider)
    assert provider.model == "gpt-extraction"


def test_azure_extraction_provider_reports_missing_configuration() -> None:
    settings = Settings(
        llm_provider="azure_openai",
        azure_openai_api_key=None,
        azure_openai_endpoint=None,
        azure_openai_api_version=None,
        azure_openai_chat_deployment=None,
        _env_file=None,
    )

    with pytest.raises(ExtractionConfigurationError, match="AZURE_OPENAI_API_KEY"):
        create_graph_extraction_provider(settings)


def test_azure_extraction_provider_uses_chat_deployment() -> None:
    settings = Settings(
        llm_provider="azure_openai",
        azure_openai_api_key="azure-secret",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_version="2025-04-01-preview",
        azure_openai_chat_deployment="enterprise-extraction",
        _env_file=None,
    )

    with patch("app.extraction.factory.AzureOpenAI") as client_class:
        provider = create_graph_extraction_provider(settings)

    client_class.assert_called_once_with(
        api_key="azure-secret",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2025-04-01-preview",
    )
    assert isinstance(provider, OpenAIGraphExtractionProvider)
    assert provider.model == "enterprise-extraction"
