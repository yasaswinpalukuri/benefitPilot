"""Hybrid retrieval mechanics, with a tiny booklet and a stub "semantic" embedder.

The stub maps words to concepts ("specs" and "glasses" both mean vision), so
it behaves like a real embedding model on the cases tested here, but is
deterministic and offline. Filler words ("plan", "cover") carry no meaning.
"""

from __future__ import annotations

import numpy as np
import pytest

from benefitpilot.retrieval.bm25 import tokenize
from benefitpilot.retrieval.chunking import chunk_markdown
from benefitpilot.retrieval.embeddings import EmbeddingError, normalize
from benefitpilot.retrieval.hybrid import HybridRetriever, RetrievalConfig

BOOKLET = """# PLAN-T — Test booklet

> Synthetic.

## §1 Vision

Glasses and contact lenses for your eyes.

| Rule | Value |
|---|---|
| Annual maximum | $200 |

## §2 Dental

Crowns, fillings and cleanings by a dentist.

## §3 General

This plan covers many services. The plan pays claims for covered services under the plan.
"""
OTHER = BOOKLET.replace("PLAN-T — Test", "PLAN-U — Other")

CONCEPTS = {
    "vision": {"glass", "lense", "eye", "specs", "spec", "vision"},
    "dental": {"crown", "filling", "cleaning", "dentist", "teeth", "dental"},
}


class StubEmbedder:
    name = "stub"

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on  # "index" or "query"

    def _vec(self, text: str) -> np.ndarray:
        toks = set(tokenize(text))
        return np.array([len(toks & words) for words in CONCEPTS.values()], dtype=np.float32)

    def embed_documents(self, texts):
        if self.fail_on == "index":
            raise EmbeddingError("server down")
        return normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text):
        if self.fail_on == "query":
            raise EmbeddingError("server down")
        return normalize(self._vec(text))


CHUNKS = chunk_markdown(BOOKLET) + chunk_markdown(OTHER)
CFG = RetrievalConfig(min_bm25=1.0, min_cosine=0.5)  # tiny corpus, small BM25 scores


def retriever(embedder=None, cfg=CFG):
    return HybridRetriever(CHUNKS, embedder if embedder is not None else StubEmbedder(), cfg)


def test_plan_filter():
    r = retriever().search("glasses", plan_id="PLAN-U")
    assert r.hits and all(h.chunk.plan_id == "PLAN-U" for h in r.hits)


def test_vectors_catch_paraphrases_bm25_misses():
    r = retriever()
    assert r.search("new specs", plan_id="PLAN-T", mode="bm25").hits == []
    assert r.search("new specs", plan_id="PLAN-T").citations[0] == "PLAN-T §1"


def test_weak_lexical_matches_do_not_outvote_vectors():
    # "plan"/"cover" match §3 lexically (weakly); the meaning is vision.
    r = retriever(cfg=RetrievalConfig(min_bm25=100.0, min_cosine=0.5))
    res = r.search("does the plan cover specs", plan_id="PLAN-T")
    assert res.citations[0] == "PLAN-T §1"
    general = next(h for h in res.hits if h.citation == "PLAN-T §3")
    assert general.bm25_rank == 1  # it was the BM25 winner, but too weak to vote


def test_rrf_score_is_sum_of_reciprocal_ranks():
    res = retriever().search("glasses lenses", plan_id="PLAN-T")
    top = res.hits[0]
    assert top.citation == "PLAN-T §1"
    assert top.bm25_rank == 1 and top.vector_rank == 1
    assert top.score == pytest.approx(2 / (60 + 1), abs=1e-6)


def test_relevance_gate_uses_cosine_when_vectors_exist():
    r = retriever()
    assert r.search("crown", plan_id="PLAN-T").relevant
    off_topic = r.search("hockey scores tonight", plan_id="PLAN-T")
    assert not off_topic.relevant


def test_no_embedder_means_bm25_only_not_degraded():
    r = HybridRetriever(CHUNKS, None, CFG)
    res = r.search("glasses", plan_id="PLAN-T")
    assert (res.mode, res.degraded, res.top_cosine) == ("bm25", None, None)
    assert res.citations[0] == "PLAN-T §1"
    assert res.relevant  # BM25 gate
    with pytest.raises(EmbeddingError):
        r.search("glasses", mode="vector")


@pytest.mark.parametrize("fail_on", ["index", "query"])
def test_embedder_failure_degrades_to_bm25(fail_on):
    res = retriever(StubEmbedder(fail_on=fail_on)).search("glasses", plan_id="PLAN-T")
    assert res.mode == "bm25"
    assert "server down" in res.degraded
    assert res.citations[0] == "PLAN-T §1"


def test_hits_are_returned_even_when_gate_fails():
    res = retriever().search("plan", plan_id="PLAN-T")  # lexical only, no concept but general
    assert res.hits  # kept for the audit log / transparency panel
