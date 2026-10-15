"""Build the retriever from settings and config/thresholds.yaml."""

from __future__ import annotations

import yaml

from benefitpilot.config import CONFIG_DIR, Settings
from benefitpilot.retrieval.chunking import load_chunks
from benefitpilot.retrieval.embeddings import (
    CachedEmbedder,
    Embedder,
    OllamaEmbedder,
)
from benefitpilot.retrieval.hybrid import HybridRetriever, RetrievalConfig


def load_retrieval_config() -> RetrievalConfig:
    raw = yaml.safe_load((CONFIG_DIR / "thresholds.yaml").read_text(encoding="utf-8"))
    return RetrievalConfig(**raw.get("retrieval", {}))


def build_embedder(settings: Settings) -> Embedder | None:
    if settings.embeddings_backend == "none":
        return None
    inner = OllamaEmbedder(
        settings.ollama_base_url,
        settings.ollama_embed_model,
        timeout_s=settings.llm_timeout_s,
    )
    return CachedEmbedder(inner, settings.cache_dir)


def build_retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(load_chunks(), build_embedder(settings), load_retrieval_config())
