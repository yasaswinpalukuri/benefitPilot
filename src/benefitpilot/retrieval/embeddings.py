"""Embedding backends.

- `OllamaEmbedder`: nomic-embed-text on the home Ollama server.
- `CachedEmbedder`: caches document vectors on disk, keyed by model and text
  hash, so restarting the API doesn't re-embed every chunk on a CPU-only box.

With EMBEDDINGS_BACKEND=none (the default, and what CI uses), retrieval is
BM25 only. See docs/DECISIONS.md, ADR-014, for why there is no offline
"fake" embedder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, *, timeout_s: float = 120.0, batch: int = 16):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.timeout_s = timeout_s
        self.batch = batch
        # nomic-embed-text is trained with task prefixes; other models are not.
        nomic = model.startswith("nomic-embed")
        self.doc_prefix = "search_document: " if nomic else ""
        self.query_prefix = "search_query: " if nomic else ""

    def _embed(self, inputs: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                for i in range(0, len(inputs), self.batch):
                    r = client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": inputs[i : i + self.batch]},
                    )
                    r.raise_for_status()
                    vectors.extend(r.json()["embeddings"])
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise EmbeddingError(f"Ollama embedding failed at {self.base_url}: {e}") from e
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        safe = "".join(ch if ch.isalnum() else "_" for ch in inner.name)
        self.path = cache_dir / f"embeddings-{safe}.json"

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.name}\n{text}".encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        cache: dict[str, list[float]] = {}
        if self.path.exists():
            cache = json.loads(self.path.read_text(encoding="utf-8"))
        keys = [self._key(t) for t in texts]
        missing = [t for t, k in zip(texts, keys, strict=True) if k not in cache]
        if missing:
            for t, v in zip(missing, self.inner.embed_documents(missing), strict=True):
                cache[self._key(t)] = v.tolist()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(cache), encoding="utf-8")
        return np.asarray([cache[k] for k in keys], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)
