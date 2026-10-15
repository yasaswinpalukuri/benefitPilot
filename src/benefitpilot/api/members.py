"""GET /members and GET /members/{member_id}/usage (synthetic members only)."""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from benefitpilot.api.deps import get_plans, get_store
from benefitpilot.rules.plan_schema import Plan
from benefitpilot.rules.summary import MemberSummary, summarize
from benefitpilot.store import MemberNotFound, MemberStore

router = APIRouter(prefix="/members", tags=["members"])


class MemberListItem(BaseModel):
    member_id: str
    name: str
    plan_id: str
    plan_name: str
    relationship: str
    primary_member_id: str | None


class MemberUsage(BaseModel):
    name: str
    summary: MemberSummary


def get_today() -> dt.date:
    return dt.date.today()


@router.get("", response_model=list[MemberListItem])
def list_members(
    plans: Annotated[dict[str, Plan], Depends(get_plans)],
    store: Annotated[MemberStore, Depends(get_store)],
) -> list[MemberListItem]:
    return [
        MemberListItem(
            member_id=m.member_id,
            name=m.name,
            plan_id=m.plan_id,
            plan_name=plans[m.plan_id].name,
            relationship=m.relationship,
            primary_member_id=m.primary_member_id,
        )
        for m in store.list_members()
    ]


@router.get("/{member_id}/usage", response_model=MemberUsage)
def member_usage(
    member_id: str,
    plans: Annotated[dict[str, Plan], Depends(get_plans)],
    store: Annotated[MemberStore, Depends(get_store)],
    today: Annotated[dt.date, Depends(get_today)],
) -> MemberUsage:
    try:
        info = store.member_info(member_id)
        snapshot = store.snapshot(member_id, today.year)
    except MemberNotFound as e:
        raise HTTPException(status_code=404, detail=f"Unknown member {member_id}") from e
    return MemberUsage(name=info.name, summary=summarize(plans[info.plan_id], snapshot, today))
