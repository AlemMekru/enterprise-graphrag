"""Graph extraction provider contract."""

from typing import Protocol

from app.models.graph import ProviderExtraction


class GraphExtractionProvider(Protocol):
    """Contract implemented by structured extraction backends."""

    def extract(self, text: str) -> ProviderExtraction:
        """Extract schema-constrained graph information from text."""
        ...
