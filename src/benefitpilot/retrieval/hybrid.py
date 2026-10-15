"""Hybrid retrieval: BM25 + vector search, fused with reciprocal-rank fusion.

RRF score = sum over retrievers of 1 / (rrf_k + rank). It uses ranks only,
so BM25 scores (unbounded) and cosines (-1..1) never need to be put on the
same scale (docs/DECISIONS.md, ADR-015).

Only BM25 hits scoring at least `min_bm25` get a vote in the fusion. Weak
lexical matches on generic words ("plan", "about") otherwise outvote a
correct vector top-1. Ties go to the better vector rank.

RRF says nothing about whether the best chunk is actually *relevant*. A
separate gate does that. With semantic vectors it uses the top cosine only,
because BM25 can't separate in-scope from out-of-scope questions (a single
generic word scores as high as a real match). Without vectors
(EMBEDDINGS_BACKEND=none, or the embedder is down) it falls back to the
top BM25 score. When `relevant` is False, callers
must answer "I couldn't find that in your plan" (CLAUDE.md, non-negotiable 3).
The hits are still returned so the audit log and the "Why this answer?"
panel can show the near-misses.

If the embedder is unavailable (e.g. the home server is offline), retrieval
degrades to BM25 only and says so in the result, rather than failing the
request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from benefitpilot.retrieval.bm25 import BM25Index
from benefitpilot.retrieval.chunking import Chunk
from benefitpilot.retrieval.embeddings import Embedder, EmbeddingError

log = logging.getLogger(__name__)
Mode = Literal["hybrid", "bm25", "vector"]


@dataclass(frozen=True)
class RetrievalConfig:
    rrf_k: int = 60
    candidates: int = 20  # per-retriever depth fed into fusion
    min_bm25: float = 4.0  # a "meaningful" lexical match: may vote in fusion / pass the gate
    min_cosine: float = 0.57  # relevance gate when semantic vectors are available


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # RRF score (or the raw score in single-retriever modes)
    bm25_rank: int | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    cosine: float | None = None

    @property
    def citation(self) -> str:
        return self.chunk.id


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    hits: list[Hit]
    relevant: bool
    mode: Mode
    top_bm25: float
    top_cosine: float | None
    degraded: str | None = None  # why the vector side was skipped, if it was
    plan_id: str | None = None

    @property
    def citations(self) -> list[str]:
        return [h.citation for h in self.hits]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder | None,
        config: RetrievalConfig | None = None,
    ):
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = chunks
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.bm25 = BM25Index([c.text for c in chunks])
        self.matrix: np.ndarray | None = None
        self.unavailable: str | None = None  # set when an embedder is configured but failed
        if embedder is not None:
            try:
                self.matrix = embedder.embed_documents([c.text for c in chunks])
            except EmbeddingError as e:
                self.unavailable = str(e)
                log.warning("Vector index disabled, using BM25 only: %s", e)

    def _mask(self, plan_id: str | None) -> np.ndarray:
        return np.array([plan_id is None or c.plan_id == plan_id for c in self.chunks])

    @staticmethod
    def _ranked(scores: np.ndarray, mask: np.ndarray, depth: int, positive: bool) -> list[int]:
        idx = [i for i in np.argsort(-scores, kind="stable") if mask[i]]
        if positive:
            idx = [i for i in idx if scores[i] > 0]
        return idx[:depth]

    def search(
        self, query: str, *, plan_id: str | None = None, k: int = 5, mode: Mode = "hybrid"
    ) -> RetrievalResult:
        cfg = self.config
        mask = self._mask(plan_id)
        bm25 = np.asarray(self.bm25.scores(query))
        top_bm25 = float(bm25[mask].max()) if mask.any() else 0.0

        cosines: np.ndarray | None = None
        degraded = None
        if mode != "bm25" and self.embedder is not None:
            if self.matrix is None:
                degraded = self.unavailable
            else:
                try:
                    cosines = self.matrix @ self.embedder.embed_query(query)
                except EmbeddingError as e:
                    degraded = str(e)
                    log.warning("Query embedding failed, using BM25 only: %s", e)
        if mode == "vector" and cosines is None:
            raise EmbeddingError(degraded or "vector search unavailable")
        effective: Mode = "bm25" if cosines is None else mode

        bm25_ranked = self._ranked(bm25, mask, cfg.candidates, positive=True)
        vec_ranked = (
            self._ranked(cosines, mask, cfg.candidates, positive=False)
            if cosines is not None
            else []
        )
        bm25_rank = {i: r for r, i in enumerate(bm25_ranked, start=1)}
        vec_rank = {i: r for r, i in enumerate(vec_ranked, start=1)}

        if effective == "bm25":
            fused = {i: float(bm25[i]) for i in bm25_ranked}
        elif effective == "vector":
            fused = {i: float(cosines[i]) for i in vec_ranked}
        else:
            voters = {i: r for i, r in bm25_rank.items() if bm25[i] >= cfg.min_bm25}
            fused = {}
            for ranks in (voters, vec_rank):
                for i, r in ranks.items():
                    fused[i] = fused.get(i, 0.0) + 1.0 / (cfg.rrf_k + r)

        inf = len(self.chunks) + 1
        order = sorted(
            fused,
            key=lambda i: (-fused[i], vec_rank.get(i, inf), bm25_rank.get(i, inf), i),
        )[:k]
        hits = [
            Hit(
                chunk=self.chunks[i],
                score=round(fused[i], 6),
                bm25_rank=bm25_rank.get(i),
                bm25_score=round(float(bm25[i]), 4) if i in bm25_rank else None,
                vector_rank=vec_rank.get(i),
                cosine=round(float(cosines[i]), 4) if cosines is not None else None,
            )
            for i in order
        ]

        top_cos = float(cosines[mask].max()) if cosines is not None and mask.any() else None
        # Cosine gate when vectors exist; BM25 can't separate in-scope from out-of-scope.
        relevant = top_cos >= cfg.min_cosine if top_cos is not None else top_bm25 >= cfg.min_bm25
        return RetrievalResult(
            query=query,
            hits=hits,
            relevant=relevant,
            mode=effective,
            top_bm25=round(top_bm25, 4),
            top_cosine=round(top_cos, 4) if top_cos is not None else None,
            degraded=degraded,
            plan_id=plan_id,
        )
