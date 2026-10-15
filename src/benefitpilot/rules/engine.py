"""Deterministic claim checking. Pure function: no I/O, no model calls, no mutation.

Order of evaluation
1. Is the service excluded, or not in the plan at all? If so, stop: no other
   rule applies to a service the plan never pays for.
2. Deny-type rules: coverage active, dependent eligibility, waiting period,
   category age limit, frequency, trip length, pre-authorization. All of them
   are evaluated and reported, not just the first failure, so the
   "Why this answer?" panel shows the complete picture.
3. If nothing denied the claim, compute the payment:
   billed -> minus deductible -> times coinsurance -> capped by per-visit,
   per-trip, annual, group and lifetime maximums (each cap a reduce-type check).

Outcome: deny if any deny-type rule failed or a cap reduced payment to $0;
partial if a cap reduced it; otherwise approve.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from benefitpilot.rules.models import (
    ClaimInput,
    ClaimResult,
    Effect,
    MemberSnapshot,
    Money,
    RuleCheck,
    Violation,
)
from benefitpilot.rules.plan_schema import Category, Plan

CENT = Decimal("0.01")
ZERO = Decimal("0.00")


def money(value: Decimal | float | int | str) -> Money:
    """Round to cents, half up, as a currency amount (never via binary float)."""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(value: Money) -> str:
    return f"${value:,.2f}"


def add_months(d: dt.date, months: int) -> dt.date:
    """Calendar months, clamping the day (Jan 31 + 1 month -> Feb 28/29)."""
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return dt.date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def age_on(dob: dt.date, on: dt.date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


class _Checks:
    def __init__(self, plan: Plan):
        self.plan = plan
        self.items: list[RuleCheck] = []

    def add(
        self,
        rule: str,
        passed: bool,
        effect: Effect,
        detail: str,
        section: str | None,
        violation: Violation | None = None,
    ) -> None:
        self.items.append(
            RuleCheck(
                rule=rule,
                passed=passed,
                effect=effect,
                detail=detail,
                citation=self.plan.citation(section) if section else None,
                violation=None if passed else violation,
            )
        )

    def failed(self, effect: Effect) -> bool:
        return any(not c.passed and c.effect == effect for c in self.items)


def check_claim(plan: Plan, member: MemberSnapshot, claim: ClaimInput) -> ClaimResult:
    if member.plan_id != plan.plan_id:
        raise ValueError(f"member {member.member_id} is on {member.plan_id}, not {plan.plan_id}")
    if member.member_id != claim.member_id:
        raise ValueError("claim and member snapshot are for different members")
    if member.benefit_year != claim.service_date.year:
        raise ValueError(
            f"usage snapshot is for {member.benefit_year}, claim is dated {claim.service_date}"
        )

    checks = _Checks(plan)
    billed = money(claim.amount)
    category = plan.categories.get(claim.category)
    excluded = {x.code: x for x in plan.exclusions.items}

    if claim.category in excluded:
        checks.add(
            "exclusion",
            False,
            "deny",
            f"{excluded[claim.category].label} is excluded under this plan.",
            plan.exclusions.section,
            Violation.EXCLUDED_SERVICE,
        )
    elif category is None:
        checks.add(
            "covered_category",
            False,
            "deny",
            f"'{claim.category}' is not a benefit category of {plan.plan_id}.",
            None,
            Violation.NOT_COVERED,
        )
    else:
        _deny_checks(checks, plan, category, member, claim)

    if category is None or checks.failed("deny"):
        return _result(plan, member, claim, checks, billed, ZERO, None, ZERO, category)

    deductible = _deductible(checks, plan, category, member, billed)
    share = money((billed - deductible) * Decimal(str(category.coinsurance)))
    covered = _apply_caps(checks, plan, category, member, claim, share)
    return _result(plan, member, claim, checks, billed, deductible, share, covered, category)


# --- deny-type rules ---------------------------------------------------------


def _deny_checks(
    checks: _Checks, plan: Plan, cat: Category, member: MemberSnapshot, claim: ClaimInput
) -> None:
    on = claim.service_date

    active = on >= member.coverage_start
    checks.add(
        "coverage_active",
        active,
        "deny",
        f"Coverage started {member.coverage_start}; service date {on}.",
        None,
        Violation.COVERAGE_NOT_ACTIVE,
    )

    if member.relationship == "child":
        e = plan.eligibility
        limit = e.student_age_limit if member.is_student else e.child_age_limit
        age = age_on(member.dob, on)
        kind = "full-time student" if member.is_student else "child"
        checks.add(
            "dependent_eligibility",
            age < limit,
            "deny",
            f"Dependent {kind} is {age} on the service date; covered under age {limit}.",
            e.section,
            Violation.DEPENDENT_INELIGIBLE_AGE,
        )

    months = plan.waiting_periods.months.get(claim.category)
    if months is not None:
        eligible_from = add_months(member.coverage_start, months)
        checks.add(
            "waiting_period",
            on >= eligible_from,
            "deny",
            f"{months}-month waiting period from coverage start {member.coverage_start}; "
            f"eligible from {eligible_from}.",
            plan.waiting_periods.section,
            Violation.WAITING_PERIOD,
        )

    if cat.age_limit is not None:
        age = age_on(member.dob, on)
        checks.add(
            "age_limit",
            age < cat.age_limit,
            "deny",
            f"Patient is {age}; {cat.label} is covered under age {cat.age_limit}.",
            cat.section,
            Violation.AGE_LIMIT_EXCEEDED,
        )

    if cat.frequency is not None:
        last = member.usage_for(claim.category).last_service_date
        window = cat.frequency.months
        if last is None:
            ok, detail = True, f"No previous paid {cat.label.lower()} on record."
        else:
            next_allowed = add_months(last, window)
            ok = on >= next_allowed
            detail = f"Last paid {last}; limit 1 per {window} months, next allowed {next_allowed}."
        checks.add("frequency", ok, "deny", detail, cat.section, Violation.FREQUENCY_LIMIT)

    if cat.max_trip_days is not None:
        if claim.trip_days is None:
            checks.add(
                "trip_info",
                False,
                "deny",
                "Trip length (trip_days) is required for travel claims.",
                cat.section,
                Violation.MISSING_INFORMATION,
            )
        else:
            checks.add(
                "trip_length",
                claim.trip_days <= cat.max_trip_days,
                "deny",
                f"Trip of {claim.trip_days} days; maximum {cat.max_trip_days} days.",
                cat.section,
                Violation.TRIP_TOO_LONG,
            )

    threshold = plan.preauthorization.thresholds.get(claim.category)
    if threshold is not None:
        required = threshold == 0 or claim.amount > Decimal(str(threshold))
        rule = "all claims" if threshold == 0 else f"claims over {fmt(money(threshold))}"
        if not required:
            ok, detail = True, f"Pre-authorization needed only for {rule}."
        elif claim.preauth_on_file:
            ok, detail = True, f"Pre-authorization required ({rule}) and on file."
        else:
            ok, detail = False, f"Pre-authorization required ({rule}) but none on file."
        checks.add(
            "preauthorization",
            ok,
            "deny",
            detail,
            plan.preauthorization.section,
            Violation.PREAUTH_MISSING,
        )


# --- payment -----------------------------------------------------------------


def _deductible(
    checks: _Checks, plan: Plan, cat: Category, member: MemberSnapshot, billed: Money
) -> Money:
    annual = money(plan.deductible.annual_per_person)
    if not cat.deductible_applies or annual == 0:
        return ZERO
    remaining = max(annual - money(member.deductible_applied), ZERO)
    applied = min(remaining, billed)
    checks.add(
        "deductible",
        True,
        "info",
        f"{fmt(applied)} of this claim goes to the {fmt(annual)} deductible "
        f"({fmt(remaining)} was left this year).",
        plan.deductible.section,
    )
    return applied


def _apply_caps(
    checks: _Checks,
    plan: Plan,
    cat: Category,
    member: MemberSnapshot,
    claim: ClaimInput,
    share: Money,
) -> Money:
    usage = member.usage_for(claim.category)
    caps: list[_Cap] = []

    if cat.per_visit_max is not None:
        limit = money(cat.per_visit_max)
        caps.append(
            _Cap(
                "per_visit_max",
                limit,
                f"{fmt(limit)} per-visit maximum",
                cat.section,
                Violation.PER_VISIT_MAX_EXCEEDED,
            )
        )
    if cat.per_trip_max is not None:
        limit = money(cat.per_trip_max)
        caps.append(
            _Cap(
                "per_trip_max",
                limit,
                f"{fmt(limit)} per-trip maximum",
                cat.section,
                Violation.PER_TRIP_MAX_EXCEEDED,
            )
        )
    if cat.annual_max is not None:
        limit, used = money(cat.annual_max), money(usage.amount_paid)
        caps.append(
            _Cap(
                "annual_max",
                max(limit - used, ZERO),
                f"{fmt(limit)} annual maximum ({fmt(used)} used this year)",
                cat.section,
                Violation.ANNUAL_MAX_EXCEEDED,
            )
        )
    group = plan.groups[cat.group] if cat.group else None
    if group is not None and group.combined_annual_max is not None:
        limit = money(group.combined_annual_max)
        in_group = [n for n, c in plan.categories.items() if c.group == cat.group]
        used = money(sum((member.usage_for(n).amount_paid for n in in_group), Decimal("0")))
        caps.append(
            _Cap(
                "group_max",
                max(limit - used, ZERO),
                f"{fmt(limit)} combined {group.label.lower()} maximum ({fmt(used)} used this year)",
                group.section,
                Violation.GROUP_MAX_EXCEEDED,
            )
        )
    if cat.lifetime_max is not None:
        limit, used = money(cat.lifetime_max), money(usage.lifetime_paid)
        caps.append(
            _Cap(
                "lifetime_max",
                max(limit - used, ZERO),
                f"{fmt(limit)} lifetime maximum ({fmt(used)} used)",
                cat.section,
                Violation.LIFETIME_MAX_EXCEEDED,
            )
        )

    covered = share
    for cap in caps:
        if covered > cap.available:
            detail = f"Plan share {fmt(covered)} reduced to {fmt(cap.available)}: {cap.label}."
            checks.add(cap.rule, False, "reduce", detail, cap.section, cap.violation)
            covered = cap.available
        else:
            detail = (
                f"Plan share {fmt(covered)} is within the {cap.label}; "
                f"{fmt(cap.available)} available."
            )
            checks.add(cap.rule, True, "reduce", detail, cap.section)
    return covered


@dataclass(frozen=True)
class _Cap:
    rule: str
    available: Money  # most the plan can still pay under this limit
    label: str
    section: str
    violation: Violation


def _result(
    plan: Plan,
    member: MemberSnapshot,
    claim: ClaimInput,
    checks: _Checks,
    billed: Money,
    deductible: Money,
    share: Money | None,
    covered: Money,
    category: Category | None,
) -> ClaimResult:
    if checks.failed("deny"):
        outcome = "deny"
    elif checks.failed("reduce"):
        outcome = "partial" if covered > 0 else "deny"
    else:
        outcome = "approve"

    citations = [plan.citation(category.section)] if category else []
    for c in checks.items:
        if not c.passed and c.citation and c.citation not in citations:
            citations.append(c.citation)

    return ClaimResult(
        member_id=member.member_id,
        plan_id=plan.plan_id,
        category=claim.category,
        service_date=claim.service_date,
        outcome=outcome,
        billed_amount=billed,
        deductible_applied=deductible,
        plan_share_before_limits=share,
        covered_amount=covered,
        member_pays=billed - covered,
        violations=[c.violation for c in checks.items if c.violation is not None],
        checks=checks.items,
        citations=citations,
    )
