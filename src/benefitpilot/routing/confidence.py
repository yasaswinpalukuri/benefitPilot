"""TypeSafe's published confidence formulas (docs.typesafe.ai/confidence).

FallbackRouter uses them so its Choice/Score confidence values sit on the
same scale as Jev's and the two are comparable in the benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence


def choice_confidence(probabilities: Sequence[float]) -> float:
    """(n * p_max - 1) / (n - 1): 0 for an even split, 1 for certainty."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return _clamp((n * max(probabilities) - 1) / (n - 1))


def score_confidence(probabilities: Sequence[float]) -> float:
    """1 - (probability-weighted distance from the modal level) / (same for a uniform spread)."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    peak = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - peak) for i, p in enumerate(probabilities))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return _clamp(1 - spread / even)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))
