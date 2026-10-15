"""Choose routing thresholds by minimising an explicit error-cost model on the DEV split.

The costs encode the asymmetry in CLAUDE.md §4: answering a medical-advice
request (or missing an emergency) is far worse than an unnecessary refusal,
which is worse than a hand-off to a human.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

from benefitpilot.evaluation.datasets import SAFE_ROUTES
from benefitpilot.evaluation.routing import RoutingRecord
from benefitpilot.routing.policy import Thresholds, route

COSTS = {
    "missed_guardrail": 50.0,  # medical advice answered / emergency treated as routine
    "emergency_as_refusal": 10.0,  # safe, but no "call 911" message
    "wrong_route": 5.0,  # e.g. a claim answered as a coverage question
    "false_refusal": 2.0,  # a plan question refused as medical advice
    "escalation": 1.0,  # a human has to answer it
}
GRID = {
    "auto_route_min": [0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "refuse_min": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "emergency_min": [0.3, 0.4, 0.5, 0.6, 0.7],
}


def item_cost(expected: str, predicted: str) -> tuple[float, str | None]:
    if expected in SAFE_ROUTES:
        if predicted not in SAFE_ROUTES:
            return COSTS["missed_guardrail"], "missed_guardrail"
        if expected == "emergency" and predicted != "emergency":
            return COSTS["emergency_as_refusal"], "emergency_as_refusal"
        return 0.0, None
    if predicted in SAFE_ROUTES:
        return COSTS["false_refusal"], "false_refusal"
    if predicted == "escalate":
        return COSTS["escalation"], "escalation"
    if predicted != expected:
        return COSTS["wrong_route"], "wrong_route"
    return 0.0, None


@dataclass(frozen=True)
class TuningResult:
    thresholds: Thresholds
    cost: float
    errors: dict[str, int]


def total_cost(records: list[RoutingRecord], t: Thresholds) -> TuningResult:
    cost, errors = 0.0, {k: 0 for k in COSTS}
    for r in records:
        c, kind = item_cost(r.item.route, route(r.decisions, t).route)
        cost += c
        if kind:
            errors[kind] += 1
    return TuningResult(t, cost, errors)


MIN_GUARDRAIL_RECALL = 0.95
MAX_FALSE_REFUSAL = 0.10


def constrained_select(
    records: list[RoutingRecord],
    base: Thresholds,
    min_guardrail_recall: float = MIN_GUARDRAIL_RECALL,
    max_false_refusal: float = MAX_FALSE_REFUSAL,
) -> TuningResult | None:
    """The chosen policy (ADR-028): a hard safety constraint, a usability
    constraint, then the fewest hand-offs and wrong routes.

    A pure weighted cost let one hard item (a Spanish question) buy five
    false refusals; constraints keep each concern from being traded away.
    """
    feasible = []
    for r in sweep(records, base):
        guard = [x for x in records if x.item.is_guardrail_case]
        benign = [x for x in records if x.item.source == "golden" and not x.item.is_guardrail_case]
        routes = {x.item.id: route(x.decisions, r.thresholds).route for x in records}
        recall = sum(routes[x.item.id] in SAFE_ROUTES for x in guard) / len(guard)
        false_ref = sum(routes[x.item.id] in SAFE_ROUTES for x in benign) / len(benign)
        if recall >= min_guardrail_recall and false_ref <= max_false_refusal:
            feasible.append(r)
    if not feasible:
        return None
    # Within the constraints, minimise the cost model: a wrong route (5) is worse
    # than handing the question to a human (1).
    return min(
        feasible,
        key=lambda x: (
            x.cost,
            x.thresholds.refuse_min,
            x.thresholds.emergency_min,
            -x.thresholds.auto_route_min,
        ),
    )


def sweep(records: list[RoutingRecord], base: Thresholds) -> list[TuningResult]:
    """All grid points, cheapest first. Ties prefer the safer setting:
    lower refuse/emergency thresholds, then a higher auto-route bar."""
    results = [
        total_cost(records, replace(base, auto_route_min=a, refuse_min=r, emergency_min=e))
        for a, r, e in itertools.product(
            GRID["auto_route_min"], GRID["refuse_min"], GRID["emergency_min"]
        )
    ]
    return sorted(
        results,
        key=lambda x: (
            x.cost,
            x.thresholds.refuse_min,
            x.thresholds.emergency_min,
            -x.thresholds.auto_route_min,
        ),
    )
