"""Retrieval metrics over eval/retrieval_set.jsonl: hit@k, MRR and gate accuracy.

A question counts as a hit@k if *any* of its acceptable citations appears in
the top k. "none" questions (out of scope) are scored only by the relevance
gate: the gate must say "not relevant".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benefitpilot.config import REPO_ROOT
from benefitpilot.retrieval.hybrid import HybridRetriever, Mode

RETRIEVAL_SET = REPO_ROOT / "eval" / "retrieval_set.jsonl"


@dataclass(frozen=True)
class Item:
    id: str
    kind: str  # lexical | paraphrase | none
    plan_id: str
    question: str
    expected: list[str]


def load_items(path: Path = RETRIEVAL_SET) -> list[Item]:
    with path.open(encoding="utf-8") as f:
        return [Item(**json.loads(line)) for line in f if line.strip()]


@dataclass(frozen=True)
class Metrics:
    n: int
    hit: dict[int, float]  # k -> fraction of answerable questions hit in top k
    mrr: float
    gate_accuracy: float  # answerable -> relevant, none -> not relevant
    misses: list[str]  # ids not hit in top max(k)


def evaluate(
    retriever: HybridRetriever,
    items: list[Item],
    *,
    mode: Mode = "hybrid",
    ks: tuple[int, ...] = (1, 3, 5),
) -> Metrics:
    answerable = [i for i in items if i.expected]
    hits = {k: 0 for k in ks}
    rr = 0.0
    gate_ok = 0
    misses = []
    for item in items:
        result = retriever.search(item.question, plan_id=item.plan_id, k=max(ks), mode=mode)
        gate_ok += result.relevant == bool(item.expected)
        if not item.expected:
            continue
        ranks = [r for r, c in enumerate(result.citations, start=1) if c in item.expected]
        first = ranks[0] if ranks else None
        for k in ks:
            hits[k] += first is not None and first <= k
        rr += 1 / first if first else 0.0
        if first is None:
            misses.append(item.id)
    n = len(answerable) or 1
    return Metrics(
        n=len(answerable),
        hit={k: hits[k] / n for k in ks},
        mrr=rr / n,
        gate_accuracy=gate_ok / (len(items) or 1),
        misses=misses,
    )
