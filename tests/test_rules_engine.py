"""Unit tests for the rules engine, using hand-built member snapshots.

Expected amounts are worked out by hand in each test, not computed with
the engine's own helpers.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from benefitpilot.rules.engine import add_months, check_claim, money
from benefitpilot.rules.models import CategoryUsage as U
from benefitpilot.rules.models import ClaimInput, MemberSnapshot, Violation
from benefitpilot.rules.plan_schema import load_plans

PLANS = load_plans()
A, B = PLANS["PLAN-A"], PLANS["PLAN-B"]
SERVICE = dt.date(2026, 10, 15)
D = Decimal


def snap(plan="PLAN-A", **kw) -> MemberSnapshot:
    defaults = dict(
        member_id="T1",
        plan_id=plan,
        dob=dt.date(1985, 3, 1),
        relationship="self",
        is_student=False,
        coverage_start=dt.date(2015, 1, 1),
        benefit_year=2026,
        deductible_applied=D("50") if plan == "PLAN-A" else D("0"),
        usage={},
    )
    return MemberSnapshot(**(defaults | kw))


def claim(category, amount, on=SERVICE, **kw) -> ClaimInput:
    return ClaimInput(member_id="T1", category=category, amount=D(str(amount)), date=on, **kw)


def run(plan, member, c):
    return check_claim(plan, member, c)


# --- payment arithmetic ------------------------------------------------------


def test_canonical_physio_example():
    """$420 of $500 used; $120 session at 80% = $96 share, capped at the $80 left."""
    r = run(A, snap(usage={"physiotherapy": U(amount_paid=D("420"))}), claim("physiotherapy", 120))
    assert r.outcome == "partial"
    assert r.plan_share_before_limits == D("96.00")
    assert r.covered_amount == D("80.00")
    assert r.member_pays == D("40.00")
    assert r.violations == [Violation.ANNUAL_MAX_EXCEEDED]
    assert r.citations == ["PLAN-A §4.2.1"]


def test_clean_claim_is_approved():
    r = run(A, snap(), claim("physiotherapy", 100))
    assert (r.outcome, r.covered_amount, r.violations) == ("approve", D("80.00"), [])


def test_deductible_partly_met():
    # $20 deductible left: (100 - 20) * 0.8 = 64
    r = run(A, snap(deductible_applied=D("30")), claim("physiotherapy", 100))
    assert r.deductible_applied == D("20.00")
    assert r.covered_amount == D("64.00")
    assert r.outcome == "approve"


def test_claim_fully_absorbed_by_deductible_is_not_a_denial():
    r = run(A, snap(deductible_applied=D("0")), claim("physiotherapy", 30))
    assert r.deductible_applied == D("30.00")
    assert r.covered_amount == D("0.00")
    assert r.outcome == "approve"
    assert r.violations == []


def test_deductible_does_not_apply_to_drugs():
    r = run(A, snap(deductible_applied=D("0")), claim("drug_tier_1", 50))
    assert r.deductible_applied == D("0.00")
    assert r.covered_amount == D("40.00")


def test_gold_has_no_deductible_and_pays_100_percent():
    r = run(B, snap("PLAN-B"), claim("physiotherapy", 100))
    assert (r.covered_amount, r.member_pays) == (D("100.00"), D("0.00"))
    assert not any(c.rule == "deductible" for c in r.checks)


def test_annual_max_exhausted_is_a_denial():
    r = run(A, snap(usage={"physiotherapy": U(amount_paid=D("500"))}), claim("physiotherapy", 80))
    assert r.outcome == "deny"
    assert r.covered_amount == D("0.00")
    assert r.violations == [Violation.ANNUAL_MAX_EXCEEDED]


def test_per_visit_max():
    # Chiro: 70 * 0.8 = 56, capped at $40 per visit.
    r = run(A, snap(), claim("chiropractic", 70))
    assert (r.outcome, r.covered_amount) == ("partial", D("40.00"))
    assert r.violations == [Violation.PER_VISIT_MAX_EXCEEDED]


def test_per_visit_then_annual_max_both_bind():
    # 56 -> 40 (per visit) -> 20 (only $20 of $300 left)
    member = snap(usage={"chiropractic": U(amount_paid=D("280"))})
    r = run(A, member, claim("chiropractic", 70))
    assert r.covered_amount == D("20.00")
    assert r.violations == [Violation.PER_VISIT_MAX_EXCEEDED, Violation.ANNUAL_MAX_EXCEEDED]


def test_dental_combined_group_max():
    # Basic + major share $1,000; $950 used. 100 * 0.8 = 80, capped at $50.
    usage = {"dental_basic": U(amount_paid=D("500")), "dental_major": U(amount_paid=D("450"))}
    r = run(A, snap(usage=usage), claim("dental_basic", 100))
    assert (r.outcome, r.covered_amount) == ("partial", D("50.00"))
    assert r.violations == [Violation.GROUP_MAX_EXCEEDED]
    assert "PLAN-A §5" in r.citations


def test_orthodontics_lifetime_max():
    # 400 * 0.5 = 200, capped at the $100 left of the $2,500 lifetime maximum.
    member = snap(
        "PLAN-B",
        relationship="child",
        dob=dt.date(2014, 1, 1),
        usage={"orthodontics": U(lifetime_paid=D("2400"))},
    )
    r = run(B, member, claim("orthodontics", 400, preauth_on_file=True))
    assert (r.outcome, r.covered_amount) == ("partial", D("100.00"))
    assert r.violations == [Violation.LIFETIME_MAX_EXCEEDED]


def test_money_rounds_half_up_in_decimal():
    # 10.05 * 0.70 = 7.035 exactly. Binary floats give 7.034999..., which rounds to 7.03.
    r = run(A, snap(), claim("drug_tier_2", "10.05"))
    assert r.covered_amount == D("7.04")
    assert money(0.1 + 0.2) == D("0.30")


def test_covered_plus_member_pays_equals_billed():
    r = run(A, snap(usage={"physiotherapy": U(amount_paid=D("420"))}), claim("physiotherapy", 120))
    assert r.covered_amount + r.member_pays == r.billed_amount


# --- deny-type rules ---------------------------------------------------------


def test_excluded_service_short_circuits():
    r = run(A, snap(), claim("orthodontics", 900))
    assert r.outcome == "deny"
    assert r.violations == [Violation.EXCLUDED_SERVICE]
    assert [c.rule for c in r.checks] == ["exclusion"]
    assert r.citations == ["PLAN-A §8"]
    assert r.plan_share_before_limits is None


def test_unknown_category_is_not_covered():
    r = run(A, snap(), claim("acupuncture", 80))
    assert r.violations == [Violation.NOT_COVERED]
    assert r.citations == []


@pytest.mark.parametrize(
    "plan, age, student, eligible",
    [
        (A, 20, False, True),
        (A, 21, False, False),  # turns 21 on the service date: "under age 21"
        (A, 21, True, True),
        (A, 24, True, True),
        (A, 25, True, False),
        (B, 21, False, True),
        (B, 22, False, False),
        (B, 25, True, True),
        (B, 26, True, False),
    ],
)
def test_dependent_age_limits(plan, age, student, eligible):
    dob = dt.date(SERVICE.year - age, SERVICE.month, SERVICE.day)
    member = snap(plan.plan_id, relationship="child", dob=dob, is_student=student)
    r = run(plan, member, claim("physiotherapy", 80))
    assert (Violation.DEPENDENT_INELIGIBLE_AGE not in r.violations) == eligible
    if not eligible:
        assert f"{plan.plan_id} §2.2" in r.citations


def test_age_limits_do_not_apply_to_spouses():
    member = snap(relationship="spouse", dob=dt.date(1950, 1, 1))
    assert run(A, member, claim("physiotherapy", 80)).outcome == "approve"


@pytest.mark.parametrize("age, ok", [(18, True), (19, False)])
def test_orthodontics_age_limit(age, ok):
    dob = dt.date(SERVICE.year - age, 1, 1)
    member = snap("PLAN-B", relationship="child", dob=dob)
    r = run(B, member, claim("orthodontics", 300, preauth_on_file=True))
    assert (Violation.AGE_LIMIT_EXCEEDED not in r.violations) == ok


@pytest.mark.parametrize(
    "last, ok",
    [
        (None, True),
        (dt.date(2025, 11, 1), False),
        (dt.date(2024, 10, 16), False),  # one day short of 24 months
        (dt.date(2024, 10, 15), True),  # exactly 24 months
    ],
)
def test_frequency_limit(last, ok):
    member = snap(usage={"eye_exam": U(last_service_date=last)})
    r = run(A, member, claim("eye_exam", 70))
    assert (r.outcome == "approve") == ok
    assert (Violation.FREQUENCY_LIMIT in r.violations) == (not ok)


@pytest.mark.parametrize(
    "start, ok",
    [(dt.date(2026, 6, 1), False), (dt.date(2025, 10, 16), False), (dt.date(2025, 10, 15), True)],
)
def test_waiting_period(start, ok):
    r = run(A, snap(coverage_start=start), claim("dental_major", 400))
    assert (Violation.WAITING_PERIOD not in r.violations) == ok
    if not ok:
        assert r.covered_amount == D("0.00")
        assert r.plan_share_before_limits is None


@pytest.mark.parametrize(
    "category, amount, preauth, ok",
    [
        ("dental_major", 600, False, False),
        ("dental_major", 600, True, True),
        ("dental_major", 500, False, True),  # threshold is "over $500"
        ("drug_tier_3", 20, False, False),  # all claims
        ("drug_tier_3", 20, True, True),
    ],
)
def test_preauthorization(category, amount, preauth, ok):
    r = run(A, snap(), claim(category, amount, preauth_on_file=preauth))
    assert (Violation.PREAUTH_MISSING not in r.violations) == ok


def test_travel_trip_length():
    assert run(A, snap(), claim("travel_emergency", 2000, trip_days=30)).covered_amount == D(
        "2000.00"
    )
    r = run(A, snap(), claim("travel_emergency", 2000, trip_days=31))
    assert r.violations == [Violation.TRIP_TOO_LONG]
    r = run(A, snap(), claim("travel_emergency", 2000))
    assert r.violations == [Violation.MISSING_INFORMATION]


def test_service_before_coverage_start():
    r = run(A, snap(coverage_start=dt.date(2026, 11, 1)), claim("physiotherapy", 80))
    assert Violation.COVERAGE_NOT_ACTIVE in r.violations


def test_all_deny_reasons_are_reported_not_just_the_first():
    member = snap(relationship="child", dob=dt.date(2000, 1, 1))  # 26, not a student
    r = run(A, member, claim("drug_tier_3", 50))
    assert r.violations == [Violation.DEPENDENT_INELIGIBLE_AGE, Violation.PREAUTH_MISSING]
    assert r.covered_amount == D("0.00")


def test_every_applicable_rule_is_logged_with_a_citation():
    r = run(A, snap(), claim("chiropractic", 30))
    rules = [c.rule for c in r.checks]
    assert rules == ["coverage_active", "deductible", "per_visit_max", "annual_max"]
    assert all(c.citation for c in r.checks if c.rule != "coverage_active")


# --- guards ------------------------------------------------------------------


def test_rejects_member_from_another_plan():
    with pytest.raises(ValueError, match="PLAN-B"):
        run(A, snap("PLAN-B"), claim("physiotherapy", 80))


def test_rejects_usage_from_another_year():
    with pytest.raises(ValueError, match="2026"):
        run(A, snap(), claim("physiotherapy", 80, on=dt.date(2027, 1, 5)))


def test_add_months_clamps_day():
    assert add_months(dt.date(2026, 1, 31), 1) == dt.date(2026, 2, 28)
    assert add_months(dt.date(2023, 2, 28), 12) == dt.date(2024, 2, 28)
    assert add_months(dt.date(2026, 11, 30), 3) == dt.date(2027, 2, 28)
