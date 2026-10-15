"""Violation detection: the rules engine against the generator's constructed labels (ADR-004)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from benefitpilot.evaluation.metrics import precision_recall
from benefitpilot.rules.engine import check_claim
from benefitpilot.rules.models import ClaimInput
from benefitpilot.rules.plan_schema import Plan
from benefitpilot.store import MemberStore


def evaluate_violations(claims: list[Any], plans: dict[str, Plan], store: MemberStore) -> dict:
    """`claims`: generator Claim objects (with expected_violation / expected_outcome)."""
    rows = []
    for c in claims:
        result = check_claim(
            plans[c.plan_id],
            store.snapshot(c.member_id, c.service_date.year),
            ClaimInput(
                member_id=c.member_id,
                category=c.category,
                amount=Decimal(str(c.amount)),
                date=c.service_date,
                preauth_on_file=c.preauth_on_file,
                trip_days=c.trip_days,
            ),
        )
        rows.append((c, result))

    has_violation = [c.expected_violation != "none" for c, _ in rows]
    flagged = [bool(r.violations) for _, r in rows]
    precision, recall = precision_recall(has_violation, flagged)

    per_type = {}
    for v in sorted({c.expected_violation for c, _ in rows} - {"none"}):
        labels = [c.expected_violation == v for c, _ in rows]
        preds = [v in [str(x) for x in r.violations] for _, r in rows]
        p, rcl = precision_recall(labels, preds)
        per_type[v] = {"n": sum(labels), "precision": p, "recall": rcl}

    outcome_correct = sum(r.outcome == c.expected_outcome for c, r in rows)
    return {
        "n_claims": len(rows),
        "n_with_violation": sum(has_violation),
        "precision": precision,
        "recall": recall,
        "outcome_accuracy": outcome_correct / len(rows),
        "per_type": per_type,
    }
