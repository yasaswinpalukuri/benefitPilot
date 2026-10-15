"""A small Okapi BM25 implementation.

Written out instead of imported so every part of the scoring is visible and
testable. With about 60 chunks, speed is not a concern.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its
    me my of on or our so that the their them there these this to was what when
    where which who will with you your""".split()  # noqa: SIM905
)


def _stem(token: str) -> str:
    """Very light plural folding: 'claims' -> 'claim', 'therapies' -> 'therapy'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    # Single letters ("what's" -> "s") are noise; single digits ("tier 3") are not.
    return [
        _stem(t)
        for t in TOKEN_RE.findall(text.lower())
        if t not in STOPWORDS and (len(t) > 1 or t.isdigit())
    ]


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        # Lucene's non-negative IDF variant.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            out.append(s)
        return out
