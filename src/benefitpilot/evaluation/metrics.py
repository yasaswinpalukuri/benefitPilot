"""Small, dependency-free metrics with exact definitions (tested in tests/test_eval_metrics.py)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def confusion_matrix(
    labels: Sequence[str], preds: Sequence[str], classes: Sequence[str]
) -> list[list[int]]:
    """rows = true class, columns = predicted class."""
    index = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for y, p in zip(labels, preds, strict=True):
        m[index[y]][index[p]] += 1
    return m


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def precision_recall(
    labels: Sequence[bool], preds: Sequence[bool]
) -> tuple[float | None, float | None]:
    tp = sum(y and p for y, p in zip(labels, preds, strict=True))
    fp = sum((not y) and p for y, p in zip(labels, preds, strict=True))
    fn = sum(y and (not p) for y, p in zip(labels, preds, strict=True))
    return rate(tp, tp + fp), rate(tp, tp + fn)


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float | None
    observed: float | None  # fraction correct / positive


@dataclass(frozen=True)
class Calibration:
    ece: float
    bins: list[CalibrationBin]
    n: int


def calibration(
    confidences: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> Calibration:
    """Expected Calibration Error with equal-width bins.

    ECE = sum over bins of (bin size / n) * |mean confidence - observed frequency|.
    For a Choice: confidence = top probability, outcome = top choice correct.
    For a Noul: confidence = P(yes), outcome = label is yes.
    """
    n = len(confidences)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)
        ]
        if idx:
            conf = sum(confidences[i] for i in idx) / len(idx)
            obs = sum(outcomes[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(conf - obs)
            bins.append(CalibrationBin(lo, hi, len(idx), conf, obs))
        else:
            bins.append(CalibrationBin(lo, hi, 0, None, None))
    return Calibration(ece=ece, bins=bins, n=n)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0-100), so the value is one that was observed."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]
