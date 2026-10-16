"""Threshold policy: turns raw System One probabilities into a route.

Pure code, so the policy is explicit, testable and re-tunable from logged
probabilities without re-running inference. Order matters, and is ordered by
the cost of getting it wrong:

1. Emergency  — P(life-threatening urgency) >= emergency_min. Checked first:
   someone describing chest pain needs "call 911", not a refusal.
2. Refuse     — asks_medical_advice >= refuse_min, or the intent model's
   P(medical_advice_request) >= refuse_min, or medical advice is the intent's
   most likely option even below that threshold. Three signals, any one
   refuses. The argmax rule uses the *comparison* between options, which an
   under-confident classifier still gets right (ADR-028): on the held-out red
   team it reached 100% recall at 9.7% false refusals, where lowering the
   threshold alone needed 22.6% false refusals for the same recall.
3. Auto-route — top intent probability >= auto_route_min.
4. Escalate   — anything less certain goes to the human-review queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import yaml

from benefitpilot.config import CONFIG_DIR
from benefitpilot.routing.questions import (
    ANSWER_CONTAINS_MEDICAL_ADVICE,
    ANSWER_SUPPORTED_BY_CONTEXT,
    ASKS_MEDICAL_ADVICE,
    INTENT,
    NEEDS_MEMBER_DATA,
    URGENCY,
)
from benefitpilot.routing.types import Decisions

Route = Literal[
    "emergency",
    "refuse_medical",
    "escalate",
    "claim_check",
    "coverage",
    "procedure",
    "out_of_scope",
]
INTENT_ROUTES: dict[str, Route] = {
    "coverage_question": "coverage",
    "claim_check": "claim_check",
    "procedure_how_to": "procedure",
    "medical_advice_request": "refuse_medical",
    "out_of_scope": "out_of_scope",
}


@dataclass(frozen=True)
class Thresholds:
    auto_route_min: float = 0.9
    refuse_min: float = 0.5
    emergency_min: float = 0.5
    output_medical_advice_max: float = 0.5
    output_grounded_min: float = 0.5

    @classmethod
    def load(cls, router: str | None = None) -> Thresholds:
        """Base thresholds (tuned for the FallbackRouter), with `routers.<router>`
        overrides on top: each System 1 model gets thresholds tuned on its own
        probabilities (ADR-029)."""
        raw = yaml.safe_load((CONFIG_DIR / "thresholds.yaml").read_text(encoding="utf-8"))
        override = (raw.get("routers") or {}).get(router or "", {}) or {}
        return cls(
            auto_route_min=override.get("auto_route_min", raw["intent"]["auto_route_min"]),
            refuse_min=override.get("refuse_min", raw["medical_advice"]["refuse_min"]),
            emergency_min=override.get("emergency_min", raw["urgency"]["emergency_min"]),
            output_medical_advice_max=raw["output_check"]["medical_advice_max"],
            output_grounded_min=raw["output_check"]["grounded_min"],
        )


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    intent: str
    intent_p: float
    needs_member_data: bool


def route(decisions: Decisions, t: Thresholds) -> RouteDecision:
    intent = decisions.choice(INTENT.id)
    p_medical = decisions.noul(ASKS_MEDICAL_ADVICE.id).p
    urgency = decisions.score(URGENCY.id)
    p_emergency = urgency.probabilities.get(len(URGENCY.criteria) - 1, 0.0)
    p_intent_medical = intent.probabilities.get("medical_advice_request", 0.0)
    needs_data = decisions.noul(NEEDS_MEMBER_DATA.id).p >= 0.5

    def decide(r: Route, reason: str) -> RouteDecision:
        return RouteDecision(r, reason, intent.choice, intent.top_p, needs_data)

    if p_emergency >= t.emergency_min:
        return decide("emergency", f"P(emergency)={p_emergency:.2f} >= {t.emergency_min}")
    medical_is_top = intent.choice == "medical_advice_request"
    if p_medical >= t.refuse_min or p_intent_medical >= t.refuse_min or medical_is_top:
        return decide(
            "refuse_medical",
            f"asks_medical_advice={p_medical:.2f}, P(intent=medical)={p_intent_medical:.2f}"
            f"{' (most likely intent)' if medical_is_top else ''}; refuse at >= {t.refuse_min}"
            " or when medical advice is the most likely intent",
        )
    if intent.top_p >= t.auto_route_min:
        return decide(
            INTENT_ROUTES[intent.choice],
            f"intent={intent.choice} p={intent.top_p:.2f} >= {t.auto_route_min}",
        )
    return decide(
        "escalate",
        f"intent={intent.choice} p={intent.top_p:.2f} < {t.auto_route_min}; needs human review",
    )


@dataclass(frozen=True)
class OutputVerdict:
    passed: bool
    reasons: list[str]


def check_output(decisions: Decisions, t: Thresholds) -> OutputVerdict:
    reasons = []
    p_advice = decisions.noul(ANSWER_CONTAINS_MEDICAL_ADVICE.id).p
    p_grounded = decisions.noul(ANSWER_SUPPORTED_BY_CONTEXT.id).p
    if p_advice >= t.output_medical_advice_max:
        reasons.append(f"answer_contains_medical_advice={p_advice:.2f}")
    if p_grounded < t.output_grounded_min:
        reasons.append(f"answer_supported_by_context={p_grounded:.2f}")
    return OutputVerdict(passed=not reasons, reasons=reasons)
