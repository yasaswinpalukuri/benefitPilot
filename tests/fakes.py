"""Test doubles: a router that returns scripted decisions and a scripted LLM."""

from __future__ import annotations

from collections.abc import Callable

from benefitpilot.llm.client import LLMError, LLMResult
from benefitpilot.routing.confidence import choice_confidence, score_confidence
from benefitpilot.routing.types import (
    ChoiceDecision,
    Decisions,
    NoulDecision,
    Question,
    RouterState,
    ScoreDecision,
)


def _choice(options: list[str], chosen: str, p: float) -> ChoiceDecision:
    rest = (1 - p) / (len(options) - 1) if len(options) > 1 else 0.0
    probs = {o: (p if o == chosen else rest) for o in options}
    return ChoiceDecision(
        choice=chosen, probabilities=probs, confidence=choice_confidence(list(probs.values()))
    )


class ScriptedRouter:
    """Answers whatever questions are asked from a small script.

    intent / intent_p, medical, urgency (3 level probabilities), member_data,
    category (claim_category option), amount (index of the $ amount in the
    message, 1-based), advice / grounded for the output check.
    """

    name = "scripted"

    def __init__(
        self,
        intent="coverage_question",
        intent_p=0.95,
        medical=0.02,
        urgency=(0.98, 0.02, 0.0),
        member_data=0.1,
        category="unclear",
        amount=1,
        advice=0.02,
        grounded=0.95,
    ):
        self.script = dict(
            intent=intent,
            intent_p=intent_p,
            medical=medical,
            urgency=urgency,
            member_data=member_data,
            category=category,
            amount=amount,
            advice=advice,
            grounded=grounded,
        )
        self.calls: list[list[str]] = []

    async def decide(self, state: RouterState, questions: list[Question]) -> Decisions:
        s = self.script
        self.calls.append([q.id for q in questions])
        answers = {}
        for q in questions:
            if q.id == "intent":
                answers[q.id] = _choice(q.options, s["intent"], s["intent_p"])
            elif q.id == "asks_medical_advice":
                answers[q.id] = NoulDecision(p=s["medical"])
            elif q.id == "urgency":
                u = list(s["urgency"])
                answers[q.id] = ScoreDecision(
                    score=sum(i * p for i, p in enumerate(u)),
                    probabilities=dict(enumerate(u)),
                    confidence=score_confidence(u),
                )
            elif q.id == "needs_member_data":
                answers[q.id] = NoulDecision(p=s["member_data"])
            elif q.id == "claim_category":
                answers[q.id] = _choice(q.options, s["category"], 0.9)
            elif q.id == "claim_amount":
                answers[q.id] = _choice(q.options, f"amount_{s['amount']}", 0.9)
            elif q.id == "answer_contains_medical_advice":
                answers[q.id] = NoulDecision(p=s["advice"])
            elif q.id == "answer_supported_by_context":
                answers[q.id] = NoulDecision(p=s["grounded"])
        return Decisions(answers=answers, router=self.name, model="scripted", latency_ms=0.1)


class FakeLLM:
    provider = "fake"
    model = "fake-llm"

    def __init__(self, reply: str | Callable[[str, str], str] = "", fail: bool = False):
        self.reply, self.fail = reply, fail
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, *, max_tokens: int = 400) -> LLMResult:
        self.calls.append((system, user))
        if self.fail:
            raise LLMError("model offline")
        text = self.reply(system, user) if callable(self.reply) else self.reply
        return LLMResult(text, self.model, 100, 20, 5.0, 0.0)
