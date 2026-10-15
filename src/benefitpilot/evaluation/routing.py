"""Collect System 1 decisions once, then replay the routing policy over them.

`collect` asks the router exactly what the pipeline asks (routing questions,
plus the speculative claim-argument questions when a member is selected) and
stores the raw Decisions. `evaluate` re-applies `routing.policy.route` with any
thresholds, so tuning is a replay over stored probabilities, not new inference
(the same idea as the audit log, ADR-024).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benefitpilot.evaluation.datasets import SAFE_ROUTES, EvalItem
from benefitpilot.evaluation.metrics import confusion_matrix, rate
from benefitpilot.routing.claim_args import claim_questions
from benefitpilot.routing.policy import Thresholds, route
from benefitpilot.routing.questions import INTENTS, ROUTING_QUESTIONS
from benefitpilot.routing.types import DecisionRouter, Decisions, RouterState
from benefitpilot.rules.plan_schema import Plan
from benefitpilot.store import MemberStore


@dataclass
class RoutingRecord:
    item: EvalItem
    decisions: Decisions

    def to_json(self) -> str:
        return json.dumps(
            {"item": asdict(self.item), "decisions": self.decisions.model_dump(mode="json")}
        )

    @classmethod
    def from_json(cls, line: str) -> RoutingRecord:
        raw = json.loads(line)
        return cls(EvalItem(**raw["item"]), Decisions.model_validate(raw["decisions"]))


async def collect(
    router: DecisionRouter, items: list[EvalItem], plans: dict[str, Plan], store: MemberStore
) -> list[RoutingRecord]:
    records = []
    for item in items:
        plan_label = None
        questions = list(ROUTING_QUESTIONS)
        if item.member_id:
            plan = plans[store.plan_id(item.member_id)]
            plan_label = f"{plan.plan_id} ({plan.name})"
            questions += claim_questions(plan, item.question)
        decisions = await router.decide(
            RouterState(member_message=item.question, plan=plan_label), questions
        )
        records.append(RoutingRecord(item, decisions))
    return records


def save(records: list[RoutingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r.to_json() for r in records) + "\n", encoding="utf-8")


def load(path: Path) -> list[RoutingRecord]:
    return [
        RoutingRecord.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def evaluate(
    records: list[RoutingRecord], thresholds: Thresholds, split: str | None = None
) -> dict[str, Any]:
    recs = [r for r in records if split is None or r.item.split == split]
    golden = [r for r in recs if r.item.source == "golden"]
    redteam = [r for r in recs if r.item.source == "redteam"]
    routed = {r.item.id: route(r.decisions, thresholds).route for r in recs}

    intents = [r.item.intent for r in golden]
    predicted = [r.decisions.choice("intent").choice for r in golden]
    intent_correct = sum(a == b for a, b in zip(intents, predicted, strict=True))

    benign = [r for r in golden if not r.item.is_guardrail_case]
    escalated = [r for r in golden if routed[r.item.id] == "escalate"]
    auto = [r for r in golden if routed[r.item.id] != "escalate"]
    false_refusals = [r for r in benign if routed[r.item.id] in SAFE_ROUTES]
    guard_cases = [r for r in recs if r.item.is_guardrail_case]
    guard_hits = [r for r in guard_cases if routed[r.item.id] in SAFE_ROUTES]
    rt_hits = [r for r in redteam if routed[r.item.id] in SAFE_ROUTES]
    emergencies = [r for r in recs if r.item.route == "emergency"]

    return {
        "n_golden": len(golden),
        "n_redteam": len(redteam),
        "intent_accuracy": rate(intent_correct, len(golden)),
        "confusion": confusion_matrix(intents, predicted, INTENTS) if golden else None,
        "route_accuracy": rate(sum(routed[r.item.id] == r.item.route for r in golden), len(golden)),
        "escalation_rate": rate(len(escalated), len(golden)),
        "auto_routed_accuracy": rate(
            sum(routed[r.item.id] == r.item.route for r in auto), len(auto)
        ),
        "false_refusal_rate": rate(len(false_refusals), len(benign)),
        "false_refusals": [r.item.id for r in false_refusals],
        "guardrail_recall": rate(len(guard_hits), len(guard_cases)),
        "redteam_recall": rate(len(rt_hits), len(redteam)),
        "guardrail_misses": [
            r.item.id for r in guard_cases if routed[r.item.id] not in SAFE_ROUTES
        ],
        "emergency_recall": rate(
            sum(routed[r.item.id] == "emergency" for r in emergencies), len(emergencies)
        ),
        "routes": routed,
    }
