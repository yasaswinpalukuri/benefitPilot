from __future__ import annotations

import pytest

from benefitpilot.retrieval.bm25 import BM25Index, tokenize


@pytest.mark.parametrize(
    "text, tokens",
    [
        ("Is physiotherapy covered?", ["physiotherapy", "covered"]),
        ("Claims and therapies", ["claim", "therapy"]),
        ("What's the plan?", ["plan"]),  # stopwords and the stray "s" dropped
        ("Tier 3 drugs", ["tier", "3", "drug"]),  # single digits kept
        ("glass bus", ["glass", "bus"]),  # no over-stemming of -ss / -us
    ],
)
def test_tokenize(text, tokens):
    assert tokenize(text) == tokens


DOCS = [
    "physiotherapy physiotherapy annual maximum",
    "massage therapy annual maximum",
    "dental crowns and bridges",
    "annual maximum annual maximum annual maximum annual maximum",
]


def test_matching_document_ranks_first():
    scores = BM25Index(DOCS).scores("physiotherapy maximum")
    assert max(range(len(DOCS)), key=scores.__getitem__) == 0


def test_rare_terms_weigh_more_than_common_ones():
    idx = BM25Index(DOCS)
    assert idx.idf["crown"] > idx.idf["annual"]


def test_no_overlap_scores_zero():
    assert BM25Index(DOCS).scores("pizza") == [0.0] * len(DOCS)


def test_term_frequency_saturates():
    # Doc 3 repeats "annual" 4x but should not beat doc 1's single rare-term match by much.
    scores = BM25Index(DOCS).scores("massage annual")
    assert scores[1] > scores[3]
