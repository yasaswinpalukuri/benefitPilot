"""Evaluation data hygiene: unique IDs, balanced splits, valid labels, no leakage."""

from __future__ import annotations

from collections import Counter

from benefitpilot.evaluation.datasets import load_golden, load_holdout, load_redteam
from benefitpilot.retrieval.chunking import load_chunks
from benefitpilot.retrieval.evaluate import load_items as load_retrieval_items
from benefitpilot.routing.fallback import load_training_examples
from benefitpilot.routing.questions import INTENTS

ROUTES = {"coverage", "claim_check", "procedure", "refuse_medical", "emergency", "out_of_scope"}
GOLDEN, REDTEAM, HOLDOUT = load_golden(), load_redteam(), load_holdout()


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def test_ids_are_unique_across_all_sets():
    ids = [i.id for i in GOLDEN + REDTEAM + HOLDOUT]
    assert len(ids) == len(set(ids))


def test_sizes_and_balanced_splits():
    assert len(GOLDEN) == 80 and len(REDTEAM) == 50 and len(HOLDOUT) == 20
    assert Counter(i.split for i in GOLDEN) == {"dev": 40, "test": 40}
    assert Counter(i.split for i in REDTEAM) == {"dev": 25, "test": 25}
    per_intent = Counter((i.intent, i.split) for i in GOLDEN)
    assert all(per_intent[(intent, s)] == 8 for intent in INTENTS for s in ("dev", "test"))
    assert {i.split for i in HOLDOUT} == {"holdout"}


def test_labels_are_valid():
    for item in GOLDEN + HOLDOUT:
        assert item.intent in INTENTS, item.id
        assert item.route in ROUTES, item.id
        assert item.medical in (0, 1) and item.urgency in (0, 1, 2), item.id
    assert {i.route for i in REDTEAM} <= {"refuse_medical", "emergency"}


def test_expected_citations_exist():
    chunk_ids = {c.id for c in load_chunks()}
    for item in GOLDEN + HOLDOUT:
        assert set(item.citations) <= chunk_ids, item.id


def test_no_question_appears_in_router_training_data():
    """The FallbackRouter is trained on its own file; an eval question appearing
    there would make the routing numbers meaningless."""
    train = {_norm(e["text"]) for e in load_training_examples()}
    leaked = [i.id for i in GOLDEN + REDTEAM + HOLDOUT if _norm(i.question) in train]
    assert leaked == []


def test_no_question_is_shared_between_eval_sets():
    """Within a set a question may repeat on purpose (the retrieval set asks the
    same question once per plan); across sets it must not."""
    sets = {
        "golden": {_norm(i.question) for i in GOLDEN},
        "redteam": {_norm(i.question) for i in REDTEAM},
        "holdout": {_norm(i.question) for i in HOLDOUT},
        "retrieval": {_norm(i.question) for i in load_retrieval_items()},
    }
    names = list(sets)
    for a_i, a in enumerate(names):
        for b in names[a_i + 1 :]:
            assert not (sets[a] & sets[b]), f"{a} and {b} share {sets[a] & sets[b]}"
