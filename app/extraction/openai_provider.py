"""OpenAI-compatible structured graph extraction provider."""

from typing import Any, Protocol

from openai import OpenAIError
from pydantic import ValidationError

from app.extraction.exceptions import (
    ExtractionProviderError,
    MalformedExtractionError,
)
from app.extraction.prompt import GRAPH_EXTRACTION_SYSTEM_PROMPT
from app.models.graph import ProviderExtraction


class _ChatCompletions(Protocol):
    def parse(self, **kwargs: object) -> Any:
        """Return a schema-parsed chat completion."""
        ...


class _ChatResource(Protocol):
    completions: _ChatCompletions


class _OpenAICompatibleClient(Protocol):
    chat: _ChatResource


class OpenAIGraphExtractionProvider:
    """Extract graph candidates with Pydantic-constrained chat output."""

    def __init__(self, client: _OpenAICompatibleClient, model: str) -> None:
        self.client = client
        self.model = model

    def extract(self, text: str) -> ProviderExtraction:
        """Request and validate one structured extraction response."""
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": GRAPH_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format=ProviderExtraction,
                temperature=0,
            )
        except OpenAIError as exc:
            raise ExtractionProviderError("Graph extraction provider failed") from exc

        try:
            message = completion.choices[0].message
        except (AttributeError, IndexError, TypeError) as exc:
            raise MalformedExtractionError(
                "Extraction provider returned no completion message"
            ) from exc

        if getattr(message, "refusal", None):
            raise ExtractionProviderError("Graph extraction request was refused")
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise MalformedExtractionError(
                "Extraction provider returned no structured result"
            )
        try:
            return ProviderExtraction.model_validate(parsed)
        except ValidationError as exc:
            raise MalformedExtractionError(
                "Extraction provider response does not match the graph schema"
            ) from exc
