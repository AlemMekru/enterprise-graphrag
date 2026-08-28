"""Environment-backed application configuration."""

from functools import lru_cache
import re
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    chunk_size: int = 1_000
    chunk_overlap: int = 200

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "neo4j"
    neo4j_vector_index_name: str = "chunk_embedding_index"
    vector_similarity_function: Literal["cosine", "euclidean"] = "cosine"
    retrieval_top_k: int = 5

    llm_provider: Literal["openai", "azure_openai"] = "openai"
    embedding_provider: Literal["openai", "azure_openai"] = "openai"
    embedding_dimension: int = 1_536
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        """Reject chunk settings that cannot make forward progress."""
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not 1 <= self.embedding_dimension <= 4_096:
            raise ValueError("EMBEDDING_DIMENSION must be between 1 and 4096")
        if not 1 <= self.retrieval_top_k <= 100:
            raise ValueError("RETRIEVAL_TOP_K must be between 1 and 100")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.neo4j_vector_index_name):
            raise ValueError(
                "NEO4J_VECTOR_INDEX_NAME must contain only letters, numbers, and "
                "underscores and cannot start with a number"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance for dependency injection."""
    return Settings()
