"""FastAPI application."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI

from benefitpilot import __version__
from benefitpilot.api import ask, claims, members
from benefitpilot.config import get_settings
from benefitpilot.routing.policy import Thresholds

app = FastAPI(
    title="BenefitPilot",
    version=__version__,
    description="Coverage answers and plan-rule checks. Synthetic data only.",
)
app.include_router(claims.router)
app.include_router(ask.router)
app.include_router(members.router)


@app.get("/thresholds")
def thresholds() -> dict[str, float]:
    """The routing policy's thresholds, so clients draw the same lines the policy uses."""
    return asdict(Thresholds.load(get_settings().router_backend))


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "router_backend": settings.router_backend,
        "llm_provider": settings.llm_provider,
    }
