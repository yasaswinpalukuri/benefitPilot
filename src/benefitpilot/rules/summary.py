"""Year-to-date usage summary for a member: what's used and what's left.

Computed in code from the same plan rules and snapshot as the engine (money
is never computed by a model), for the UI's member panel.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from pydantic import BaseModel

from benefitpilot.rules.engine import ZERO, add_months, age_on, money
from benefitpilot.rules.models import MemberSnapshot, Money
from benefitpilot.rules.plan_schema import Plan


class CategorySummary(BaseModel):
    category: str
    label: str
    citation: str
    coinsurance: float
    used: Money
    annual_max: Money | None
    remaining: Money | None
    last_service_date: dt.date | None
    next_eligible: dt.date | None  # frequency-limited services only
    waiting_until: dt.date | None  # still inside a waiting period


class GroupSummary(BaseModel):
    group: str
    label: str
    citation: str
    used: Money
    combined_max: Money
    remaining: Money


class MemberSummary(BaseModel):
    member_id: str
    plan_id: str
    plan_name: str
    benefit_year: int
    relationship: str
    age: int
    coverage_start: dt.date
    dependent_eligible: bool
    deductible_annual: Money
    deductible_remaining: Money
    categories: list[CategorySummary]
    groups: list[GroupSummary]


def summarize(plan: Plan, member: MemberSnapshot, today: dt.date) -> MemberSummary:
    age = age_on(member.dob, today)
    eligible = True
    if member.relationship == "child":
        limit = (
            plan.eligibility.student_age_limit
            if member.is_student
            else plan.eligibility.child_age_limit
        )
        eligible = age < limit

    categories = []
    for name, cat in plan.categories.items():
        usage = member.usage_for(name)
        used = money(usage.amount_paid)
        annual = money(cat.annual_max) if cat.annual_max is not None else None
        next_eligible = None
        if cat.frequency and usage.last_service_date:
            candidate = add_months(usage.last_service_date, cat.frequency.months)
            next_eligible = candidate if candidate > today else None
        wait = plan.waiting_periods.months.get(name)
        waiting_until = None
        if wait is not None:
            end = add_months(member.coverage_start, wait)
            waiting_until = end if end > today else None
        categories.append(
            CategorySummary(
                category=name,
                label=cat.label,
                citation=plan.citation(cat.section),
                coinsurance=cat.coinsurance,
                used=used,
                annual_max=annual,
                remaining=max(annual - used, ZERO) if annual is not None else None,
                last_service_date=usage.last_service_date,
                next_eligible=next_eligible,
                waiting_until=waiting_until,
            )
        )

    groups = []
    for gname, group in plan.groups.items():
        if group.combined_annual_max is None:
            continue
        used = money(
            sum(
                (c.used for c in categories if plan.categories[c.category].group == gname),
                Decimal("0"),
            )
        )
        cap = money(group.combined_annual_max)
        groups.append(
            GroupSummary(
                group=gname,
                label=group.label,
                citation=plan.citation(group.section),
                used=used,
                combined_max=cap,
                remaining=max(cap - used, ZERO),
            )
        )

    annual_ded = money(plan.deductible.annual_per_person)
    return MemberSummary(
        member_id=member.member_id,
        plan_id=plan.plan_id,
        plan_name=plan.name,
        benefit_year=member.benefit_year,
        relationship=member.relationship,
        age=age,
        coverage_start=member.coverage_start,
        dependent_eligible=eligible,
        deductible_annual=annual_ded,
        deductible_remaining=max(annual_ded - money(member.deductible_applied), ZERO),
        categories=categories,
        groups=groups,
    )
