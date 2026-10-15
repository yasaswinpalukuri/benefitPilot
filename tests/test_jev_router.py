"""JevRouter against a mocked HTTP transport.

The real SDK builds and parses every request; only the network is replaced.
No API key or network access is needed (CLAUDE.md, non-negotiable 6). Response
shapes follow docs.typesafe.ai/api.
"""

from __future__ import annotations

import asyncio
import json

import httpx2
import pytest
from pydantic import SecretStr
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from benefitpilot.config import Settings
from benefitpilot.routing.factory import FailoverRouter, build_router
from benefitpilot.routing.fallback import FallbackRouter
from benefitpilot.routing.jev import USD_PER_INPUT_TOKEN, JevRouter
from benefitpilot.routing.policy import Thresholds, route
from benefitpilot.routing.questions import ROUTING_QUESTIONS
from benefitpilot.routing.types import RouterError, RouterState

ANSWERS = {
    "intent": {
        "type": "choice",
        "choice": "claim_check",
        "probabilities": {
            "coverage_question": 0.03,
            "claim_check": 0.95,
            "procedure_how_to": 0.01,
            "medical_advice_request": 0.0,
            "out_of_scope": 0.01,
        },
        "confidence": 0.94,
    },
    "asks_medical_advice": {"type": "noul", "noul": 0.02},
    "urgency": {
        "type": "score",
        "score": 0.04,
        "legend": {"0": "routine", "1": "soon", "2": "emergency"},
        "probabilities": {"0": 0.96, "1": 0.04, "2": 0.0},
        "confidence": 0.94,
    },
    "needs_member_data": {"type": "noul", "noul": 0.97},
}
OK = {
    "model": "jev-1.13.0",
    "answers": ANSWERS,
    "usage": {"input_tokens": 812, "output_tokens": 40},
}
STATE = RouterState(
    member_message="I've used $420 of physio. Will a $120 session be covered?",
    plan="PLAN-A (Bronze)",
)
FAST_RETRY = RetryPolicy(max_retries=2, backoff_initial=0.0, respect_retry_after=False)


def make_router(responses: list[tuple[int, dict]]):
    """A JevRouter whose HTTP calls return `responses` in order. Records requests."""
    calls: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        auth = request.headers.get("authorization")
        calls.append({"url": str(request.url), "auth": auth, "body": json.loads(request.content)})
        status, body = responses[min(len(calls), len(responses)) - 1]
        return httpx2.Response(status, json=body)

    client = AsyncTypeSafeClient(
        api_key="test-key",
        model="jev-1.13.0",
        retry=FAST_RETRY,
        transport=httpx2.MockTransport(handler),
        compression=False,
    )
    return JevRouter(client=client), calls


def run(router, state=STATE, questions=ROUTING_QUESTIONS):
    return asyncio.run(router.decide(state, questions))


def test_request_shape_and_pinned_model():
    router, calls = make_router([(200, OK)])
    run(router)
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["auth"] == "Bearer test-key"
    body = call["body"]
    assert body["model"] == "jev-1.13.0"  # never an alias
    assert body["state"] == {"member_message": STATE.member_message, "plan": "PLAN-A (Bronze)"}
    assert {k: v["type"] for k, v in body["questions"].items()} == {
        "intent": "choice",
        "asks_medical_advice": "noul",
        "urgency": "score",
        "needs_member_data": "noul",
    }
    assert "out_of_scope" in body["questions"]["intent"]["criteria"]  # no-match option
    assert len(body["questions"]["urgency"]["criteria"]) == 3


def test_answers_are_mapped_and_costed():
    router, _ = make_router([(200, OK)])
    d = run(router)
    assert (d.router, d.model, d.input_tokens) == ("jev", "jev-1.13.0", 812)
    assert d.cost_usd == pytest.approx(812 * USD_PER_INPUT_TOKEN)
    assert d.choice("intent").top_p == 0.95
    assert d.score("urgency").probabilities == {0: 0.96, 1: 0.04, 2: 0.0}
    r = route(d, Thresholds())
    assert (r.route, r.needs_member_data) == ("claim_check", True)


def test_decisions_round_trip_through_json_for_the_audit_log():
    router, _ = make_router([(200, OK)])
    d = run(router)
    assert type(d).model_validate_json(d.model_dump_json()) == d


def test_rate_limit_is_retried():
    router, calls = make_router([(429, {"error": "rate limited"}), (200, OK)])
    assert run(router).choice("intent").choice == "claim_check"
    assert len(calls) == 2


def test_overload_retries_then_fails():
    router, calls = make_router([(529, {"error": "overloaded"})])
    with pytest.raises(RouterError, match="Jev request failed"):
        run(router)
    assert len(calls) == 3  # initial + 2 retries


def test_auth_error_is_not_retried():
    router, calls = make_router([(401, {"error": "invalid key"})])
    with pytest.raises(RouterError, match="Authentication"):
        run(router)
    assert len(calls) == 1


def test_failover_to_fallback_is_labelled():
    jev, _ = make_router([(401, {"error": "invalid key"})])
    d = asyncio.run(FailoverRouter(jev, FallbackRouter()).decide(STATE, ROUTING_QUESTIONS))
    assert d.router == "fallback (jev failed)"
    assert "Authentication" in d.error
    assert set(d.answers) == {q.id for q in ROUTING_QUESTIONS}


def test_build_router():
    assert build_router(Settings(router_backend="fallback")).name == "fallback"
    with pytest.raises(RouterError, match="TYPESAFE_API_KEY"):
        build_router(Settings(router_backend="jev", typesafe_api_key=None))
    r = build_router(Settings(router_backend="jev", typesafe_api_key=SecretStr("k")))
    assert isinstance(r, FailoverRouter)
    assert r.primary.model == "jev-1.13.0"
