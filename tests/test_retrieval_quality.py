"""Retrieval quality on eval/retrieval_set.jsonl against the real booklets.

CI runs BM25 only (no network). The floors sit just under the measured
values, so a regression fails the build. The Ollama test runs only with
RUN_OLLAMA_TESTS=1 and checks the numbers recorded in ADR-015.
"""

from __future__ import annotations

import os

import pytest

from benefitpilot.config import get_settings
from benefitpilot.retrieval.chunking import load_chunks
from benefitpilot.retrieval.evaluate import evaluate, load_items
from benefitpilot.retrieval.factory import build_retriever, load_retrieval_config
from benefitpilot.retrieval.hybrid import HybridRetriever

ITEMS = load_items()
CHUNK_IDS = {c.id for c in load_chunks()}


def _subset(kind):
    return [i for i in ITEMS if i.kind in (kind, "none")]


def test_retrieval_set_references_real_sections():
    for item in ITEMS:
        assert set(item.expected) <= CHUNK_IDS, item.id
        assert all(e.startswith(item.plan_id) for e in item.expected), item.id


@pytest.fixture(scope="module")
def bm25_only():
    return HybridRetriever(load_chunks(), None, load_retrieval_config())


def test_bm25_lexical_floor(bm25_only):
    m = evaluate(bm25_only, _subset("lexical"))  # measured: hit@3 1.00, MRR 0.93
    assert m.hit[3] >= 0.95
    assert m.mrr >= 0.85


def test_bm25_overall_floor(bm25_only):
    m = evaluate(bm25_only, ITEMS)  # measured: hit@3 0.86
    assert m.hit[3] >= 0.80


@pytest.mark.skipif(not os.getenv("RUN_OLLAMA_TESTS"), reason="needs the Ollama server")
def test_hybrid_with_ollama():
    settings = get_settings().model_copy(update={"embeddings_backend": "ollama"})
    r = build_retriever(settings)
    assert r.matrix is not None, r.unavailable
    m = evaluate(r, ITEMS, mode="hybrid")  # measured: hit@3 1.00, gate 1.00
    assert m.hit[3] >= 0.95
    assert m.gate_accuracy >= 0.95
