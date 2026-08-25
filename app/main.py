"""FastAPI application entry point."""

from fastapi import FastAPI

from app import __version__
from app.models.health import HealthResponse

app = FastAPI(
    title="Enterprise GraphRAG",
    description="Production-oriented GraphRAG API for enterprise knowledge.",
    version=__version__,
)


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health() -> HealthResponse:
    """Report API process health."""
    return HealthResponse(status="ok", service="enterprise-graphrag")
