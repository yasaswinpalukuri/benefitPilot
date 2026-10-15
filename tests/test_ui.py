"""The Streamlit UI, its API client and its charts.

The fake API replays *real* responses captured from the FastAPI app (with a
scripted router and LLM), so the UI is tested against the true response shapes.
"""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api_client
import charts
from benefitpilot.api.app import app
from benefitpilot.api.ask import get_audit, get_pipeline
from benefitpilot.api.deps import get_store
from benefitpilot.api.members import get_today
from benefitpilot.audit.log import AuditStore
from benefitpilot.config import REPO_ROOT
from benefitpilot.graph.pipeline import Deps, Pipeline
from benefitpilot.retrieval.chunking import load_chunks
from benefitpilot.retrieval.factory import load_retrieval_config
from benefitpilot.retrieval.hybrid import HybridRetriever
from benefitpilot.routing.policy import Thresholds
from benefitpilot.rules.plan_schema import load_plans
from tests.fakes import FakeLLM, ScriptedRouter

APP_FILE = str(REPO_ROOT / "ui" / "streamlit_app.py")
CLAIM_Q = "I've used $420 of physio this year. Will a $120 session be covered?"
CLAIM_A = "The plan will pay $80.00 and you will pay $40.00 [PLAN-A §4.2.1]."
COVER_Q = "Is physiotherapy covered under my plan?"
COVER_A = "Yes, physiotherapy is covered at 80% up to $500 a year [PLAN-A §4.2.1]."


@pytest.fixture(scope="module")
def recorded(tmp_path_factory, store, canonical_claim):
    """Run the real API once per path and record its JSON."""
    audit = AuditStore(tmp_path_factory.mktemp("ui") / "audit.db")
    member = canonical_claim.member_id

    def pipeline_for(router, answer):
        return Pipeline(
            Deps(
                router=router,
                retriever=HybridRetriever(load_chunks(), None, load_retrieval_config()),
                llm=FakeLLM(answer),
                store=store,
                plans=load_plans(),
                thresholds=Thresholds(),
                audit=audit,
                today=lambda: dt.date(2026, 10, 15),
            )
        )

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_audit] = lambda: audit
    app.dependency_overrides[get_today] = lambda: dt.date(2026, 10, 15)
    client = TestClient(app)
    out = {
        "members": client.get("/members").json(),
        "usage": client.get(f"/members/{member}/usage").json(),
        "thresholds": client.get("/thresholds").json(),
        "health": client.get("/health").json(),
        "member": member,
    }
    claim_router = ScriptedRouter(intent="claim_check", category="physiotherapy", amount=2)
    for key, router, q, a in [
        ("claim", claim_router, CLAIM_Q, CLAIM_A),
        ("coverage", ScriptedRouter(), COVER_Q, COVER_A),
        ("emergency", ScriptedRouter(urgency=(0.0, 0.1, 0.9), medical=0.9), "my chest hurts", ""),
    ]:
        app.dependency_overrides[get_pipeline] = lambda r=router, ans=a: pipeline_for(r, ans)
        resp = client.post("/ask", json={"question": q, "member_id": member}).json()
        out[key] = resp
        out[key + "_audit"] = client.get(f"/audit/{resp['audit_id']}").json()
    out["escalations"] = client.get("/escalations").json()
    app.dependency_overrides.clear()
    return out


class FakeAPI:
    recorded: dict = {}
    asked: list = []

    def __init__(self, *a, **kw):
        pass

    def health(self):
        return self.recorded["health"]

    def thresholds(self):
        return self.recorded["thresholds"]

    def members(self):
        return self.recorded["members"]

    def usage(self, member_id):
        return self.recorded["usage"]

    def ask(self, question, member_id):
        FakeAPI.asked.append((question, member_id))
        key = {CLAIM_Q: "claim", COVER_Q: "coverage"}.get(question, "emergency")
        return self.recorded[key]

    def audit(self, audit_id):
        for key in ("claim", "coverage", "emergency"):
            if self.recorded[key]["audit_id"] == audit_id:
                return self.recorded[key + "_audit"]
        raise api_client.APIError("404")

    def escalations(self):
        return self.recorded["escalations"]


@pytest.fixture
def ui(monkeypatch, recorded):
    FakeAPI.recorded, FakeAPI.asked = recorded, []
    monkeypatch.setattr(api_client, "BenefitPilotAPI", FakeAPI)
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


ESCAPED_DOLLAR = chr(92) + "$"  # the UI escapes $ so Streamlit doesn't render LaTeX


def _plain(text: str) -> str:
    return text.replace(ESCAPED_DOLLAR, "$")


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [m.label + " " + str(m.value) for m in at.metric]
    return _plain("\n".join(parts))


def _button(at, text):
    return next(b for b in at.button if _plain(b.label) == text)


def test_money_is_escaped_so_it_never_renders_as_latex(ui):
    """Unescaped, "$420 ... $120" in a label rendered as a LaTeX formula (seen live)."""
    raw = [m.value for m in ui.markdown] + [b.label for b in ui.button]
    for text in raw:
        assert "$" not in text.replace(ESCAPED_DOLLAR, ""), text


def test_app_loads_with_member_list_and_disclaimer(ui, recorded):
    assert len(ui.sidebar.selectbox[0].options) == len(recorded["members"]) + 1
    assert "Synthetic data only" in _all_text(ui)
    assert not ui.chat_message  # empty chat shows example buttons
    assert len(ui.button) >= 4


def test_selecting_a_member_shows_ytd_usage(ui, recorded):
    label = next(o for o in ui.sidebar.selectbox[0].options if o.startswith(recorded["member"]))
    ui.sidebar.selectbox[0].select(label).run()
    progress_text = [_plain(p.proto.text) for p in ui.sidebar.get("progress")]
    assert any("Physiotherapy: $420.00 of $500.00 · $80.00 left" in t for t in progress_text)


def test_claim_answer_has_why_panel_with_rule_computation(ui, recorded):
    label = next(o for o in ui.sidebar.selectbox[0].options if o.startswith(recorded["member"]))
    ui.sidebar.selectbox[0].select(label).run()
    _button(ui, CLAIM_Q).click().run()
    assert not ui.exception, ui.exception
    assert FakeAPI.asked == [(CLAIM_Q, recorded["member"])]
    assert [e.label for e in ui.expander] == ["Why this answer?"]
    text = _all_text(ui)
    assert CLAIM_A in text
    assert "Claim check · rules engine" in text
    assert "Plan pays $80.00" in text and "You pay $40.00" in text
    assert "Rules engine (deterministic, no model)" in text
    assert "service `physiotherapy`" in text
    rules = ui.dataframe[0].value
    assert "annual_max" in set(rules["Rule"]) and "↓ reduced" in set(rules["Result"])
    assert len(ui.get("vega_lite_chart")) == 4  # intent, urgency, yes/no, output check


def test_coverage_answer_shows_retrieved_sections(ui):
    _button(ui, COVER_Q).click().run()
    text = _all_text(ui)
    assert "Retrieved plan sections" in text
    assert "PLAN-A §4.2.1 Physiotherapy" in text
    assert "relevance gate on top BM25" in text


def test_emergency_shows_route_and_no_llm(ui):
    ui.chat_input[0].set_value("my chest hurts").run()
    text = _all_text(ui)
    assert "Emergency · safety message" in text
    assert "call 911" in text
    assert "out tokens" not in text  # no LLM call on this path, so no LLM trace line
    assert "Rules engine" not in text and "Retrieved plan sections" not in text


def test_escalation_tab_renders(ui):
    assert [t.label for t in ui.tabs] == ["Chat", "Escalation queue"]


# --- API client ------------------------------------------------------------------


def test_api_client_errors_are_readable():
    def handler(request):
        return httpx.Response(404, json={"detail": "Unknown member M999"})

    client = api_client.BenefitPilotAPI("http://x", transport=httpx.MockTransport(handler))
    with pytest.raises(api_client.APIError, match="404: Unknown member M999"):
        client.usage("M999")


def test_api_client_sends_question_and_member():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    client = api_client.BenefitPilotAPI("http://x", transport=httpx.MockTransport(handler))
    assert client.ask("Is physio covered?", "M021") == {"answer": "ok"}
    assert seen == {"question": "Is physio covered?", "member_id": "M021"}


def test_api_client_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = api_client.BenefitPilotAPI("http://x", transport=httpx.MockTransport(handler))
    with pytest.raises(api_client.APIError, match="Cannot reach"):
        client.health()


# --- charts ------------------------------------------------------------------------


def test_charts_encode_thresholds_and_values(recorded):
    routing = recorded["claim"]["decisions"]["routing"]
    spec = charts.intent_chart(routing["intent"], 0.9).to_dict()
    flat = json.dumps(spec)
    assert "auto-route" in flat and "chosen" in flat
    u = charts.urgency_chart(routing["urgency"], 0.5).to_dict()
    assert "Possibly life-threatening" in json.dumps(u)
    n = charts.noul_chart([("asks_medical_advice", 0.02, 0.5, "refuse ≥ 50%")], "x").to_dict()
    assert "refuse" in json.dumps(n)
