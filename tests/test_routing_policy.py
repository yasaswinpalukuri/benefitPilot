from __future__ import annotations

import pytest
import yaml

from benefitpilot.config import CONFIG_DIR
from benefitpilot.routing.confidence import choice_confidence, score_confidence
from benefitpilot.routing.policy import Thresholds, check_output, route
from benefitpilot.routing.questions import INTENTS
from benefitpilot.routing.types import ChoiceDecision, Decisions, NoulDecision, ScoreDecision

T = Thresholds()


def decisions(
    intent="coverage_question",
    intent_p=0.95,
    medical=0.05,
    urgency=(0.95, 0.05, 0.0),
    member_data=0.1,
    rest=None,
) -> Decisions:
    others = [i for i in INTENTS if i != intent]
    probs = {intent: intent_p, **{o: (1 - intent_p) / len(others) for o in others}}
    if rest:
        probs.update(rest)
    return Decisions(
        answers={
            "intent": ChoiceDecision(
                choice=max(probs, key=probs.get),
                probabilities=probs,
                confidence=choice_confidence(list(probs.values())),
            ),
            "asks_medical_advice": NoulDecision(p=medical),
            "urgency": ScoreDecision(
                score=sum(i * p for i, p in enumerate(urgency)),
                probabilities=dict(enumerate(urgency)),
                confidence=score_confidence(urgency),
            ),
            "needs_member_data": NoulDecision(p=member_data),
        },
        router="test",
        latency_ms=1.0,
    )


# --- confidence formulas: worked examples from docs.typesafe.ai/confidence ---


def test_choice_confidence_examples():
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)
    assert choice_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert choice_confidence([0.9, 0.025, 0.025, 0.025, 0.025]) == pytest.approx(0.875)


def test_score_confidence_examples():
    assert score_confidence([0.0, 0.5, 0.5]) == pytest.approx(0.25)
    assert score_confidence([0.5, 0.0, 0.5]) == pytest.approx(0.0)
    assert score_confidence([0.0, 0.57, 0.43]) == pytest.approx(1 - 0.43 / (2 / 3))


# --- routing policy -----------------------------------------------------------


def test_confident_intent_routes_automatically():
    r = route(decisions("claim_check", 0.93, member_data=0.9), T)
    assert (r.route, r.needs_member_data) == ("claim_check", True)


@pytest.mark.parametrize(
    "intent, expected",
    [
        ("coverage_question", "coverage"),
        ("procedure_how_to", "procedure"),
        ("out_of_scope", "out_of_scope"),
    ],
)
def test_intent_to_route(intent, expected):
    assert route(decisions(intent, 0.95), T).route == expected


def test_uncertain_intent_escalates_to_human():
    r = route(decisions("coverage_question", 0.6), T)
    assert r.route == "escalate"
    assert "0.60" in r.reason


def test_medical_advice_noul_refuses_even_with_confident_coverage_intent():
    r = route(decisions("coverage_question", 0.95, medical=0.55), T)
    assert r.route == "refuse_medical"


def test_medical_intent_probability_alone_refuses():
    d = decisions("coverage_question", 0.45, medical=0.1, rest={"medical_advice_request": 0.5})
    assert route(d, T).route == "refuse_medical"


def test_emergency_beats_refusal():
    d = decisions("medical_advice_request", 0.95, medical=0.99, urgency=(0.0, 0.3, 0.7))
    r = route(d, T)
    assert r.route == "emergency"


def test_emergency_threshold_uses_top_level_probability_not_score():
    # Score 1.0 could be all on "soon" or split between routine and emergency.
    split = decisions(urgency=(0.45, 0.0, 0.55))
    assert route(split, T).route == "emergency"
    middle = decisions(urgency=(0.0, 1.0, 0.0))
    assert route(middle, T).route == "coverage"


def test_thresholds_load_from_yaml():
    raw = yaml.safe_load((CONFIG_DIR / "thresholds.yaml").read_text(encoding="utf-8"))
    t = Thresholds.load()
    assert t.auto_route_min == raw["intent"]["auto_route_min"]
    assert t.refuse_min == raw["medical_advice"]["refuse_min"]
    assert t.emergency_min == raw["urgency"]["emergency_min"]


def test_per_router_thresholds_override_the_base():
    raw = yaml.safe_load((CONFIG_DIR / "thresholds.yaml").read_text(encoding="utf-8"))
    jev = raw["routers"]["jev"]
    t = Thresholds.load("jev")
    assert (t.auto_route_min, t.refuse_min, t.emergency_min) == (
        jev["auto_route_min"],
        jev["refuse_min"],
        jev["emergency_min"],
    )
    # Unconfigured routers (and no router) get the base values.
    assert Thresholds.load("fallback") == Thresholds.load() == Thresholds.load("unknown")
    assert t.output_grounded_min == Thresholds.load().output_grounded_min


def test_medical_as_most_likely_intent_refuses_below_threshold():
    """The argmax rule (ADR-028): an under-confident classifier still ranks options right."""
    d = decisions(
        "medical_advice_request",
        0.35,
        medical=0.3,
        rest={"coverage_question": 0.3, "claim_check": 0.15, "out_of_scope": 0.1},
    )
    assert d.choice("intent").choice == "medical_advice_request"
    r = route(d, T)
    assert r.route == "refuse_medical"
    assert "most likely intent" in r.reason


# --- output check -------------------------------------------------------------


def out(advice, grounded) -> Decisions:
    return Decisions(
        answers={
            "answer_contains_medical_advice": NoulDecision(p=advice),
            "answer_supported_by_context": NoulDecision(p=grounded),
        },
        router="test",
        latency_ms=1.0,
    )


def test_output_check():
    assert check_output(out(0.1, 0.9), T).passed
    v = check_output(out(0.7, 0.2), T)
    assert not v.passed
    assert len(v.reasons) == 2
