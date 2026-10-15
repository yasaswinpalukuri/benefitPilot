"""Choose the router from ROUTER_BACKEND=jev|fallback."""

from __future__ import annotations

import logging

from benefitpilot.config import Settings
from benefitpilot.routing.fallback import FallbackRouter
from benefitpilot.routing.jev import JevRouter
from benefitpilot.routing.types import (
    DecisionRouter,
    Decisions,
    Question,
    RouterError,
    RouterState,
)

log = logging.getLogger(__name__)


class FailoverRouter:
    """Try the primary router; if it fails after its own retries, use the backup.

    The result says so (`router="fallback (jev failed)"`, `error=...`), so the
    audit log shows exactly which model made each decision.
    """

    def __init__(self, primary: DecisionRouter, backup: DecisionRouter):
        self.primary, self.backup = primary, backup
        self.name = primary.name

    async def decide(self, state: RouterState, questions: list[Question]) -> Decisions:
        try:
            return await self.primary.decide(state, questions)
        except RouterError as e:
            log.warning("%s failed, using %s: %s", self.primary.name, self.backup.name, e)
            decisions = await self.backup.decide(state, questions)
            return decisions.model_copy(
                update={
                    "router": f"{self.backup.name} ({self.primary.name} failed)",
                    "error": str(e),
                }
            )


def build_router(settings: Settings) -> DecisionRouter:
    fallback = FallbackRouter()
    if settings.router_backend == "fallback":
        return fallback
    if settings.typesafe_api_key is None:
        raise RouterError(
            "ROUTER_BACKEND=jev needs TYPESAFE_API_KEY (or use ROUTER_BACKEND=fallback)"
        )
    jev = JevRouter(api_key=settings.typesafe_api_key.get_secret_value(), model=settings.jev_model)
    return FailoverRouter(jev, fallback)
