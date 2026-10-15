"""FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI

from benefitpilot import __version__
from benefitpilot.api import ask, claims
from benefitpilot.config import get_settings

app = FastAPI(
    title="BenefitPilot",
    version=__version__,
    description="Coverage answers and plan-rule checks. Synthetic data only.",
)
app.include_router(claims.router)
app.include_router(ask.router)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "router_backend": settings.router_backend,
        "llm_provider": settings.llm_provider,
    }
