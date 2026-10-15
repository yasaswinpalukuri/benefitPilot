"""Compare BM25, vector and hybrid retrieval on eval/retrieval_set.jsonl.

    python scripts/eval_retrieval.py                  # uses EMBEDDINGS_BACKEND from .env
    python scripts/eval_retrieval.py --backend none   # BM25 only, offline

Prints a Markdown table per question kind (lexical, paraphrase, all).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benefitpilot.config import get_settings  # noqa: E402
from benefitpilot.retrieval.evaluate import evaluate, load_items  # noqa: E402
from benefitpilot.retrieval.factory import build_retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "none"])
    args = parser.parse_args()
    settings = get_settings()
    if args.backend:
        settings = settings.model_copy(update={"embeddings_backend": args.backend})

    retriever = build_retriever(settings)
    items = load_items()
    name = retriever.embedder.name if retriever.embedder else "none (BM25 only)"
    print(f"Embedder: {name}  ({len(items)} questions)\n")
    modes = ("bm25", "vector", "hybrid") if retriever.matrix is not None else ("bm25",)
    for kind in ("lexical", "paraphrase", "all"):
        subset = [i for i in items if kind == "all" or i.kind in (kind, "none")]
        print(f"### {kind}\n")
        print("| Mode | hit@1 | hit@3 | hit@5 | MRR | Gate accuracy | Misses (top 5) |")
        print("|---|---|---|---|---|---|---|")
        for mode in modes:
            m = evaluate(retriever, subset, mode=mode)
            print(
                f"| {mode} | {m.hit[1]:.2f} | {m.hit[3]:.2f} | {m.hit[5]:.2f} | {m.mrr:.2f} "
                f"| {m.gate_accuracy:.2f} | {', '.join(m.misses) or '-'} |"
            )
        print()


if __name__ == "__main__":
    main()
