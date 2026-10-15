"""POST /claims/check: rules engine only, no LLM, no writes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from benefitpilot.api.deps import get_plans, get_store
from benefitpilot.rules.engine import check_claim
from benefitpilot.rules.models import ClaimInput, ClaimResult
from benefitpilot.rules.plan_schema import Plan
from benefitpilot.store import MemberNotFound, MemberStore

router = APIRouter(prefix="/claims", tags=["claims"])


@router.post("/check", response_model=ClaimResult)
def check(
    claim: ClaimInput,
    plans: Annotated[dict[str, Plan], Depends(get_plans)],
    store: Annotated[MemberStore, Depends(get_store)],
) -> ClaimResult:
    try:
        member = store.snapshot(claim.member_id, claim.service_date.year)
    except MemberNotFound as e:
        raise HTTPException(status_code=404, detail=f"Unknown member {claim.member_id}") from e

    plan = plans[member.plan_id]
    known = set(plan.categories) | {x.code for x in plan.exclusions.items}
    if claim.category not in known:
        # A typo is a client error, not a "not covered" answer.
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Unknown category {claim.category!r} for {plan.plan_id}",
                "valid_categories": sorted(known),
            },
        )
    return check_claim(plan, member, claim)
