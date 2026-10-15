"""FastAPI application.

Phase 1 exposes only /health so `docker compose up` has something to run;
the real endpoints arrive in Phases 2 and 5.
"""

from __future__ import annotations

from fastapi import FastAPI

from benefitpilot import __version__
from benefitpilot.config import get_settings

app = FastAPI(
    title="BenefitPilot",
    version=__version__,
    description="Coverage answers and plan-rule checks. Synthetic data only.",
)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "router_backend": settings.router_backend,
        "llm_provider": settings.llm_provider,
    }
