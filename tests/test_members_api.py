from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from benefitpilot.api.app import app
from benefitpilot.api.deps import get_store
from benefitpilot.api.members import get_today
from benefitpilot.config import get_settings
from benefitpilot.routing.policy import Thresholds


@pytest.fixture
def client(store):
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_today] = lambda: dt.date(2026, 10, 15)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_list_members(client, dataset):
    rows = client.get("/members").json()
    assert len(rows) == len(dataset.people)
    assert {"member_id", "name", "plan_id", "plan_name", "relationship"} <= set(rows[0])
    assert {r["plan_name"] for r in rows} == {"Bronze", "Gold"}


def test_usage_matches_the_snapshot(client, canonical_claim):
    body = client.get(f"/members/{canonical_claim.member_id}/usage").json()
    s = body["summary"]
    physio = next(c for c in s["categories"] if c["category"] == "physiotherapy")
    amounts = (physio["used"], physio["annual_max"], physio["remaining"])
    assert amounts == ("420.00", "500.00", "80.00")
    assert physio["citation"] == "PLAN-A §4.2.1"
    assert s["deductible_remaining"] == "0.00"
    assert {g["group"] for g in s["groups"]} == {"drugs", "dental"}


def test_waiting_period_is_shown_for_new_hires(client, dataset):
    waiting = next(c for c in dataset.claims if c.expected_violation == "waiting_period")
    s = client.get(f"/members/{waiting.member_id}/usage").json()["summary"]
    assert any(c["waiting_until"] for c in s["categories"])


def test_ineligible_dependent_is_flagged(client, dataset):
    claim = next(c for c in dataset.claims if c.expected_violation == "dependent_ineligible_age")
    s = client.get(f"/members/{claim.member_id}/usage").json()["summary"]
    assert s["dependent_eligible"] is False


def test_unknown_member_usage_is_404(client):
    assert client.get("/members/M999/usage").status_code == 404


def test_thresholds_endpoint(client):
    """The UI draws the thresholds of the router actually in use (ADR-029)."""
    active = Thresholds.load(get_settings().router_backend)
    assert client.get("/thresholds").json() == active.__dict__
