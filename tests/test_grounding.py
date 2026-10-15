"""Deterministic grounding guards, including regressions from live runs."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from benefitpilot.guardrails.grounding import (
    cited_ids,
    claim_inconsistencies,
    unsupported_numbers,
)
from benefitpilot.routing.fallback import FallbackRouter
from benefitpilot.routing.questions import OUTPUT_CHECK_QUESTIONS
from benefitpilot.routing.types import RouterState

D = Decimal

# qwen2.5:7b's actual draft for the canonical claim in the Phase 5 live run.
# Every number in it is real, but the conclusion is reversed: the engine
# says $80.00 WILL be paid for this session.
QWEN_REVERSED = (
    "Based on the information provided, your plan has already paid $80 towards your "
    "physiotherapy sessions for the year, as you have already used $420 out of the annual "
    "maximum of $500 [PLAN-A §4.2.1]. Therefore, a new session would not be covered under "
    "this benefit."
)
FACTS = (
    "Decision for this claim: partly covered\nAmount billed for this claim: $120.00\n"
    "Amount the plan will pay for this claim: $80.00\nAmount the member will pay for this "
    "claim: $40.00\nRule [PLAN-A §4.2.1]: Plan share $96.00 reduced to $80.00: $500.00 annual "
    "maximum ($420.00 used this year)."
)


def test_numbers_compare_by_value_not_string():
    assert unsupported_numbers("you get $80 back, 80 %", "pays $80.00 at 80%") == []
    assert unsupported_numbers("$1,000 limit", "maximum $1000.00") == []
    assert unsupported_numbers("you get $96 back", FACTS) == []  # $96.00 is in the facts
    assert unsupported_numbers("you get $100 back", FACTS) == ["$100.00"]


def test_reversed_conclusion_passes_number_check_but_not_consistency():
    assert unsupported_numbers(QWEN_REVERSED, FACTS) == []  # why numbers alone aren't enough
    problems = claim_inconsistencies(QWEN_REVERSED, "partial", D("80.00"), D("40.00"))
    assert any("says not covered" in p for p in problems)
    assert any("member pays" in p for p in problems)


@pytest.mark.parametrize(
    "answer, outcome, ok",
    [
        ("The plan will pay $80.00 and you will pay $40.00.", "partial", True),
        ("It won't be covered: the plan pays $0.00 and you pay $120.00.", "deny", True),
        ("This claim is denied. Plan pays $0.00; you pay $120.00.", "deny", True),
        ("The plan pays $0.00 and you pay $120.00.", "deny", False),  # never says no
        ("Covered! The plan pays $80.00 and you pay $40.00.", "deny", False),
        ("It would not be paid. Plan $80.00, you $40.00.", "partial", False),
        ("The plan will pay $80.00.", "partial", False),  # member share missing
    ],
)
def test_claim_consistency(answer, outcome, ok):
    covered, pays = (D("80.00"), D("40.00")) if outcome != "deny" else (D("0.00"), D("120.00"))
    assert (claim_inconsistencies(answer, outcome, covered, pays) == []) == ok


def test_only_given_citations_count():
    allowed = ["PLAN-A §4.2.1", "PLAN-A §7"]
    answer = "See [PLAN-A §7] and [PLAN-A §4.2.1], also [PLAN-A §9] and [PLAN-A §7]."
    assert cited_ids(answer, allowed) == ["PLAN-A §7", "PLAN-A §4.2.1"]


def test_fallback_output_check_catches_invented_durations():
    router = FallbackRouter()
    context = "Major dental — Waiting period: 12 months"

    def grounded(answer):
        state = RouterState(
            member_message="q", assistant_answer=answer, retrieved_plan_text=context
        )
        d = asyncio.run(router.decide(state, OUTPUT_CHECK_QUESTIONS))
        return d.noul("answer_supported_by_context").p >= 0.5

    assert grounded("Major dental has a 12 month waiting period.") is True  # same duration
    assert grounded("Major dental has a waiting period of 12 months.") is True
    assert grounded("Major dental has a waiting period of 6 months.") is False
