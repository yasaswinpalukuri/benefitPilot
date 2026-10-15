"""The LangGraph pipeline, path by path, with scripted System 1 and System 2."""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal

import pytest

from benefitpilot.audit.log import AuditStore
from benefitpilot.graph.pipeline import Deps, Pipeline
from benefitpilot.guardrails import messages
from benefitpilot.retrieval.chunking import load_chunks
from benefitpilot.retrieval.factory import load_retrieval_config
from benefitpilot.retrieval.hybrid import HybridRetriever
from benefitpilot.routing.fallback import FallbackRouter
from benefitpilot.routing.policy import Thresholds
from benefitpilot.rules.plan_schema import load_plans
from benefitpilot.store import MemberNotFound
from tests.fakes import FakeLLM, ScriptedRouter

TODAY = dt.date(2026, 10, 15)
CANONICAL_Q = "I've used $420 of physio this year. Will a $120 session be covered?"
GOOD_CLAIM_ANSWER = (
    "Your plan would pay $80.00 of the $120.00 session and you would pay $40.00, because only "
    "$80.00 is left of your $500.00 physiotherapy maximum [PLAN-A §4.2.1]."
)


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(load_chunks(), None, load_retrieval_config())


@pytest.fixture
def make(store, retriever, tmp_path):
    def _make(router=None, llm=None):
        deps = Deps(
            router=router or ScriptedRouter(),
            retriever=retriever,
            llm=llm or FakeLLM(),
            store=store,
            plans=load_plans(),
            thresholds=Thresholds(),
            audit=AuditStore(tmp_path / "audit.db"),
            today=lambda: TODAY,
        )
        return Pipeline(deps)

    return _make


@pytest.fixture
def member(canonical_claim):
    return canonical_claim.member_id  # Bronze, $420 of $500 physio used


def ask(pipeline, question, member_id=None):
    return asyncio.run(pipeline.ask(question, member_id))


# --- System-1-only paths: no LLM call ------------------------------------------


@pytest.mark.parametrize(
    "script, route, text",
    [
        (dict(urgency=(0.0, 0.2, 0.8), medical=0.9), "emergency", messages.EMERGENCY),
        (dict(medical=0.7), "refuse_medical", messages.REFUSE_MEDICAL),
        (dict(intent="out_of_scope"), "out_of_scope", messages.OUT_OF_SCOPE),
    ],
)
def test_canned_routes_never_call_the_llm(make, script, route, text):
    llm = FakeLLM("should not be used")
    state, record = ask(make(ScriptedRouter(**script), llm), "question")
    assert (record.route, record.answer, record.citations) == (route, text, [])
    assert llm.calls == []


def test_uncertain_intent_is_escalated_to_the_queue(make):
    p = make(ScriptedRouter(intent_p=0.6))
    state, record = ask(p, "something ambiguous")
    assert record.route == "escalate"
    assert record.audit_id[:8] in record.answer
    queue = p.deps.audit.escalations()
    assert [e.audit_id for e in queue] == [record.audit_id]
    assert "0.60" in queue[0].reason


# --- claim path ----------------------------------------------------------------


def claim_router(**kw):
    return ScriptedRouter(intent="claim_check", category="physiotherapy", amount=2, **kw)


def test_claim_check_uses_the_rules_engine(make, member):
    llm = FakeLLM(GOOD_CLAIM_ANSWER)
    router = claim_router()
    state, record = ask(make(router, llm), CANONICAL_Q, member)
    assert record.route == "claim_check"
    assert record.answer == GOOD_CLAIM_ANSWER
    assert record.rule_result["covered_amount"] == "80.00"
    assert record.rule_result["outcome"] == "partial"
    assert record.citations == ["PLAN-A §4.2.1"]
    assert record.claim_args == {
        "category": "physiotherapy",
        "amount": "120",
        "category_p": 0.9,
        "amount_p": 0.9,
    }
    # One routing request carried the speculative claim questions; one output check.
    assert router.calls[0][-2:] == ["claim_category", "claim_amount"]
    assert len(router.calls) == 2
    # The LLM saw the engine's numbers, not the member's arithmetic.
    assert "Amount the plan will pay for this claim: $80.00" in llm.calls[0][1]


def test_money_guard_replaces_invented_amounts(make, member):
    llm = FakeLLM("Good news, you'll get $96 back and $24 is on you [PLAN-A §4.2.1].")
    state, record = ask(make(claim_router(), llm), CANONICAL_Q, member)
    assert record.answer.startswith("Based on your plan, this $120.00 physiotherapy claim")
    assert "$80.00" in record.answer
    assert any(g.startswith("money_guard") for g in record.guardrails)


def test_llm_failure_on_a_claim_falls_back_to_the_template(make, member):
    state, record = ask(make(claim_router(), FakeLLM(fail=True)), CANONICAL_Q, member)
    assert record.answer.startswith("Based on your plan")
    assert any("llm_failed" in g for g in record.guardrails)


def test_claim_without_member_asks_for_one(make):
    state, record = ask(make(claim_router()), CANONICAL_Q)
    assert record.answer == messages.NEED_MEMBER


def test_claim_with_unclear_service_asks_for_details(make, member):
    router = ScriptedRouter(intent="claim_check", category="unclear")
    state, record = ask(make(router), "Will my $60 thing be covered?", member)
    assert record.answer == messages.NEED_CLAIM_DETAILS
    assert record.rule_result is None


def test_ungrounded_claim_answer_falls_back_to_template(make, member):
    router = claim_router(grounded=0.1)
    state, record = ask(make(router, FakeLLM(GOOD_CLAIM_ANSWER)), CANONICAL_Q, member)
    assert record.answer.startswith("Based on your plan")
    assert record.route == "claim_check"


# --- coverage path -------------------------------------------------------------


def test_coverage_answer_with_citation(make, member):
    answer = "Yes. Physiotherapy is covered at 80% up to $500 a year [PLAN-A §4.2.1]."
    state, record = ask(
        make(llm=FakeLLM(answer)), "Is physiotherapy covered under my plan?", member
    )
    assert record.route == "coverage"
    assert record.answer == answer
    assert record.citations == ["PLAN-A §4.2.1"]
    assert record.retrieval["relevant"] is True
    assert all(h["citation"].startswith("PLAN-A") for h in record.retrieval["hits"])


def test_answer_without_citations_lists_retrieved_sections(make, member):
    state, record = ask(
        make(llm=FakeLLM("Yes, it is covered.")), "Is physiotherapy covered?", member
    )
    assert record.citations == [h["citation"] for h in record.retrieval["hits"]]
    assert any("no valid inline citation" in g for g in record.guardrails)


def test_question_the_plan_does_not_answer(make, member):
    llm = FakeLLM("should not be used")
    state, record = ask(make(llm=llm), "Can you recommend a good pizza place nearby?", member)
    assert record.answer == messages.NOT_FOUND
    assert llm.calls == []


def test_output_check_turns_medical_advice_into_a_refusal(make, member):
    llm = FakeLLM("Physio is covered [PLAN-A §4.2.1]. Also take 400 mg ibuprofen first.")
    state, record = ask(make(ScriptedRouter(advice=0.9), llm), "Is physiotherapy covered?", member)
    assert record.route == "refuse_medical"
    assert record.answer == messages.REFUSE_MEDICAL
    assert record.output_check["passed"] is False


def test_ungrounded_coverage_answer_is_escalated(make, member):
    p = make(ScriptedRouter(grounded=0.1), FakeLLM("Yes, unlimited physio [PLAN-A §4.2.1]."))
    state, record = ask(p, "Is physiotherapy covered?", member)
    assert record.route == "escalate"
    assert p.deps.audit.escalations()[0].reason == "answer not supported by the plan text"


def test_llm_failure_on_coverage_escalates(make, member):
    state, record = ask(make(llm=FakeLLM(fail=True)), "Is physiotherapy covered?", member)
    assert record.route == "escalate"
    assert "LLM unavailable" in record.escalation_reason


# --- audit ---------------------------------------------------------------------


def test_audit_record_is_complete_and_replayable(make, member):
    p = make(claim_router(), FakeLLM(GOOD_CLAIM_ANSWER))
    _, record = ask(p, CANONICAL_Q, member)
    stored = p.deps.audit.get(record.audit_id)
    assert stored == record
    assert stored.router["decisions"]["intent"]["probabilities"]["claim_check"] == 0.9 or (
        stored.router["decisions"]["intent"]["probabilities"]["claim_check"] > 0.9
    )
    assert set(stored.latency_ms) >= {"route", "rules", "generate", "output_check", "total"}
    assert stored.llm["prompt_sha256"] and stored.llm["draft"] == GOOD_CLAIM_ANSWER
    assert stored.output_check["decisions"]["answer_supported_by_context"]["p"] == 0.95


def test_unknown_member(make):
    with pytest.raises(MemberNotFound):
        ask(make(), "Is physio covered?", "M999")


# --- end to end with the real fallback router ----------------------------------


def test_canonical_question_end_to_end_with_fallback_router(make, member):
    state, record = ask(make(FallbackRouter(), FakeLLM(GOOD_CLAIM_ANSWER)), CANONICAL_Q, member)
    assert record.route == "claim_check"
    assert Decimal(record.rule_result["covered_amount"]) == Decimal("80.00")
    assert record.answer == GOOD_CLAIM_ANSWER
