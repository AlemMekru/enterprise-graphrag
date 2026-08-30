"""Graph extraction provider construction from shared LLM settings."""

from openai import AzureOpenAI, OpenAI

from app.config import Settings
from app.extraction.base import GraphExtractionProvider
from app.extraction.exceptions import ExtractionConfigurationError
from app.extraction.openai_provider import OpenAIGraphExtractionProvider


def create_graph_extraction_provider(settings: Settings) -> GraphExtractionProvider:
    """Build an OpenAI-compatible provider from existing LLM configuration."""
    if settings.llm_provider == "openai":
        if settings.openai_api_key is None:
            raise ExtractionConfigurationError("OPENAI_API_KEY is required")
        client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        return OpenAIGraphExtractionProvider(
            client=client,
            model=settings.openai_chat_model,
        )

    missing = [
        name
        for name, value in {
            "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
            "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
            "AZURE_OPENAI_API_VERSION": settings.azure_openai_api_version,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": settings.azure_openai_chat_deployment,
        }.items()
        if value is None
    ]
    if missing:
        raise ExtractionConfigurationError(
            f"Missing Azure OpenAI extraction configuration: {', '.join(missing)}"
        )

    assert settings.azure_openai_api_key is not None
    assert settings.azure_openai_endpoint is not None
    assert settings.azure_openai_api_version is not None
    assert settings.azure_openai_chat_deployment is not None
    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key.get_secret_value(),
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    return OpenAIGraphExtractionProvider(
        client=client,
        model=settings.azure_openai_chat_deployment,
    )
