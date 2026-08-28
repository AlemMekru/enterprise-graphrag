"""Embedding provider construction from environment settings."""

from openai import AzureOpenAI, OpenAI

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.exceptions import EmbeddingConfigurationError
from app.embeddings.openai_provider import OpenAIEmbeddingProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured embedding provider with validated credentials."""
    if settings.embedding_provider == "openai":
        if settings.openai_api_key is None:
            raise EmbeddingConfigurationError("OPENAI_API_KEY is required")
        client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        return OpenAIEmbeddingProvider(
            client=client,
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
        )

    missing = [
        name
        for name, value in {
            "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
            "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
            "AZURE_OPENAI_API_VERSION": settings.azure_openai_api_version,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": (
                settings.azure_openai_embedding_deployment
            ),
        }.items()
        if value is None
    ]
    if missing:
        raise EmbeddingConfigurationError(
            f"Missing Azure OpenAI embedding configuration: {', '.join(missing)}"
        )

    assert settings.azure_openai_api_key is not None
    assert settings.azure_openai_endpoint is not None
    assert settings.azure_openai_api_version is not None
    assert settings.azure_openai_embedding_deployment is not None
    client = AzureOpenAI(
        api_key=settings.azure_openai_api_key.get_secret_value(),
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    return OpenAIEmbeddingProvider(
        client=client,
        model=settings.azure_openai_embedding_deployment,
        dimension=settings.embedding_dimension,
    )
