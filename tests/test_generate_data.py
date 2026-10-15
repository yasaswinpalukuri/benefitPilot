"""Synthetic data: reproducible, the right size, and labels that mean what they say.

The label checks below are a small, independent re-statement of a few rules.
They are not the rules engine; they only confirm the generator built each
claim the way its label claims.
"""

from __future__ import annotations

import sqlite3
from collections import Counter

import pytest

from benefitpilot.rules.plan_schema import load_plans
from scripts.generate_data import (
    N_PER_VIOLATION,
    N_PRIMARY,
    VIOLATIONS,
    add_months,
    age_on,
    generate,
    write_sqlite,
)

PLANS = load_plans()


@pytest.fixture(scope="module")
def ds():
    return generate()


@pytest.fixture(scope="module")
def people(ds):
    return {p.member_id: p for p in ds.people}


def _dump(path) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return list(con.iterdump())
    finally:
        con.close()


def test_same_seed_same_database(tmp_path):
    write_sqlite(generate(42), tmp_path / "a.db")
    write_sqlite(generate(42), tmp_path / "b.db")
    assert _dump(tmp_path / "a.db") == _dump(tmp_path / "b.db")


def test_different_seed_different_data():
    assert generate(1).claims != generate(2).claims


def test_sizes(ds):
    assert sum(p.relationship == "self" for p in ds.people) == N_PRIMARY
    assert len(ds.claims) == 100
    assert len({c.claim_id for c in ds.claims}) == 100


def test_every_violation_type_is_represented(ds):
    counts = Counter(c.expected_violation for c in ds.claims)
    for v in VIOLATIONS:
        assert counts[v] == N_PER_VIOLATION, v
    assert counts["none"] >= 50


def test_canonical_demo_claim(ds):
    claim = next(c for c in ds.claims if c.notes.startswith("canonical"))
    assert (claim.plan_id, claim.category, claim.amount) == ("PLAN-A", "physiotherapy", 120.0)
    assert ds.usage[(claim.member_id, "physiotherapy")].amount_paid == 420.0


def test_claims_match_member_plan(ds, people):
    for c in ds.claims:
        assert people[c.member_id].plan_id == c.plan_id


def _eligible(person, plan, on) -> bool:
    if person.relationship != "child":
        return True
    limit = (
        plan.eligibility.student_age_limit
        if person.is_student
        else plan.eligibility.child_age_limit
    )
    return age_on(person.dob, on) < limit


def test_dependent_age_labels(ds, people):
    for c in ds.claims:
        ok = _eligible(people[c.member_id], PLANS[c.plan_id], c.service_date)
        assert ok == (c.expected_violation != "dependent_ineligible_age"), c


def test_exclusion_labels(ds):
    for c in ds.claims:
        excluded = c.category in {x.code for x in PLANS[c.plan_id].exclusions.items}
        assert excluded == (c.expected_violation == "excluded_service"), c


def test_waiting_period_labels(ds, people):
    for c in ds.claims:
        months = PLANS[c.plan_id].waiting_periods.months.get(c.category)
        if months is None:
            continue
        waiting = c.service_date < add_months(people[c.member_id].coverage_start, months)
        assert waiting == (c.expected_violation == "waiting_period"), c


def test_preauth_labels(ds):
    for c in ds.claims:
        threshold = PLANS[c.plan_id].preauthorization.thresholds.get(c.category)
        required = threshold is not None and (threshold == 0 or c.amount > threshold)
        missing = required and not c.preauth_on_file
        assert missing == (c.expected_violation == "preauth_missing"), c


def test_clean_claims_stay_under_per_visit_and_annual_max(ds):
    for c in ds.claims:
        if c.expected_violation != "none":
            continue
        cat = PLANS[c.plan_id].categories[c.category]
        plan_share = c.amount * cat.coinsurance
        if cat.per_visit_max is not None:
            assert plan_share <= cat.per_visit_max + 0.01, c
        if cat.annual_max is not None:
            used = ds.usage.get((c.member_id, c.category))
            assert (used.amount_paid if used else 0) + plan_share <= cat.annual_max, c


def test_database_is_marked_synthetic(tmp_path):
    path = tmp_path / "x.db"
    write_sqlite(generate(), path)
    con = sqlite3.connect(path)
    try:
        value = con.execute("SELECT value FROM meta WHERE key = 'synthetic'").fetchone()[0]
    finally:
        con.close()
    assert value.startswith("true")
