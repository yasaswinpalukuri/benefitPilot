"""Member store, plus the engine run over every generated claim.

The second part checks the engine against labels the generator assigned by
construction (ADR-004). It is the Phase 7 violation-detection metric in
miniature: any disagreement is either an engine bug or a generator bug.
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal

import pytest

from benefitpilot.rules.engine import check_claim
from benefitpilot.rules.models import ClaimInput
from benefitpilot.rules.plan_schema import load_plans
from benefitpilot.store import MemberNotFound, MemberStore

PLANS = load_plans()


def test_snapshot_reads_usage(store, canonical_claim):
    s = store.snapshot(canonical_claim.member_id, 2026)
    assert s.plan_id == "PLAN-A"
    assert s.usage["physiotherapy"].amount_paid == Decimal("420.0")
    assert s.deductible_applied == Decimal("50.0")


def test_new_benefit_year_resets_amounts_but_keeps_history(store, dataset):
    mid, cat = next(k for k, u in dataset.usage.items() if u.amount_paid)
    s = store.snapshot(mid, 2027)
    assert s.usage[cat].amount_paid == Decimal("0")
    assert s.deductible_applied == Decimal("0")

    mid, cat = next(k for k, u in dataset.usage.items() if u.last_service_date)
    s = store.snapshot(mid, 2027)
    assert s.usage[cat].last_service_date == dataset.usage[(mid, cat)].last_service_date


def test_unknown_member(store):
    with pytest.raises(MemberNotFound):
        store.snapshot("NOPE", 2026)


def test_store_is_read_only(store):
    with store._connect() as con, pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("DELETE FROM usage")


def test_missing_database_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_data.py"):
        MemberStore(tmp_path / "missing.db")


def _check(store, c):
    member = store.snapshot(c.member_id, c.service_date.year)
    return check_claim(
        PLANS[c.plan_id],
        member,
        ClaimInput(
            member_id=c.member_id,
            category=c.category,
            amount=Decimal(str(c.amount)),
            date=c.service_date,
            preauth_on_file=c.preauth_on_file,
            trip_days=c.trip_days,
        ),
    )


def test_engine_agrees_with_every_constructed_label(store, dataset):
    mismatches = []
    for c in dataset.claims:
        r = _check(store, c)
        expected = [] if c.expected_violation == "none" else [c.expected_violation]
        if [str(v) for v in r.violations] != expected or r.outcome != c.expected_outcome:
            mismatches.append((c.claim_id, c.notes, expected, r.violations, r.outcome))
    assert mismatches == []


def test_money_invariants_on_every_generated_claim(store, dataset):
    for c in dataset.claims:
        r = _check(store, c)
        assert Decimal("0") <= r.covered_amount <= r.billed_amount
        assert r.covered_amount + r.member_pays == r.billed_amount
        if r.outcome == "deny":
            assert r.covered_amount == 0
