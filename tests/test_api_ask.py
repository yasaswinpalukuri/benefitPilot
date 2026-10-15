from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from benefitpilot.api.app import app
from benefitpilot.api.ask import get_audit, get_pipeline
from benefitpilot.audit.log import AuditStore
from benefitpilot.graph.pipeline import Deps, Pipeline
from benefitpilot.retrieval.chunking import load_chunks
from benefitpilot.retrieval.factory import load_retrieval_config
from benefitpilot.retrieval.hybrid import HybridRetriever
from benefitpilot.routing.policy import Thresholds
from benefitpilot.rules.plan_schema import load_plans
from tests.fakes import FakeLLM, ScriptedRouter

ANSWER = "Your plan would pay $80.00 and you would pay $40.00 [PLAN-A §4.2.1]."
QUESTION = "I've used $420 of physio this year. Will a $120 session be covered?"


@pytest.fixture
def api(store, tmp_path):
    audit = AuditStore(tmp_path / "audit.db")
    routers = {"router": ScriptedRouter(intent="claim_check", category="physiotherapy", amount=2)}

    def pipeline():
        return Pipeline(
            Deps(
                router=routers["router"],
                retriever=HybridRetriever(load_chunks(), None, load_retrieval_config()),
                llm=FakeLLM(ANSWER),
                store=store,
                plans=load_plans(),
                thresholds=Thresholds(),
                audit=audit,
                today=lambda: dt.date(2026, 10, 15),
            )
        )

    app.dependency_overrides[get_pipeline] = pipeline
    app.dependency_overrides[get_audit] = lambda: audit
    try:
        yield TestClient(app), routers
    finally:
        app.dependency_overrides.clear()


def test_ask_claim(api, canonical_claim):
    client, _ = api
    r = client.post("/ask", json={"member_id": canonical_claim.member_id, "question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "claim_check"
    assert body["answer"] == ANSWER
    assert body["citations"] == ["PLAN-A §4.2.1"]
    assert body["rule_result"]["covered_amount"] == "80.00"
    assert body["decisions"]["routing"]["intent"]["choice"] == "claim_check"
    assert "answer_supported_by_context" in body["decisions"]["output_check"]
    assert body["latency_ms"] > 0
    assert body["cost_usd"] == 0.0

    audit = client.get(f"/audit/{body['audit_id']}")
    assert audit.status_code == 200
    assert audit.json()["rule_result"]["outcome"] == "partial"


def test_escalations_endpoint(api):
    client, routers = api
    routers["router"] = ScriptedRouter(intent_p=0.5)
    r = client.post("/ask", json={"question": "hmm, what about the thing?"})
    assert r.json()["route"] == "escalate"
    queue = client.get("/escalations").json()
    assert [e["audit_id"] for e in queue] == [r.json()["audit_id"]]
    assert queue[0]["status"] == "open"


def test_unknown_member_is_404(api):
    client, _ = api
    r = client.post("/ask", json={"member_id": "M999", "question": QUESTION})
    assert r.status_code == 404


def test_unknown_audit_id_is_404(api):
    client, _ = api
    assert client.get("/audit/nope").status_code == 404


@pytest.mark.parametrize(
    "body", [{"question": ""}, {"member_id": "M001"}, {"question": "x" * 2001}]
)
def test_ask_validation(api, body):
    client, _ = api
    assert client.post("/ask", json=body).status_code == 422
