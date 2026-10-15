"""FastAPI dependencies. Tests override these to point at a temporary database."""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException

from benefitpilot.config import get_settings
from benefitpilot.rules.plan_schema import Plan, load_plans
from benefitpilot.store import MemberStore


@lru_cache
def get_plans() -> dict[str, Plan]:
    return load_plans()


def get_store() -> MemberStore:
    try:
        return MemberStore(get_settings().db_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
