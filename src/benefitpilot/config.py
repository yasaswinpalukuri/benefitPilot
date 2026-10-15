"""Runtime settings, read from environment variables / `.env`.

Keys are optional on purpose: the project must run end to end with
ROUTER_BACKEND=fallback and no network access (CLAUDE.md, non-negotiable 6).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
PLAN_RULES_DIR = CONFIG_DIR / "plans"
PLAN_DOCS_DIR = REPO_ROOT / "data" / "plans"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    router_backend: Literal["jev", "fallback"] = "fallback"
    typesafe_api_key: SecretStr | None = None
    jev_model: str = "jev-1.13.0"

    llm_provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"
    llm_timeout_s: float = 300.0
    anthropic_api_key: SecretStr | None = None

    # none: BM25-only retrieval, no network (tests/CI default).
    embeddings_backend: Literal["ollama", "none"] = "none"

    db_path: Path = REPO_ROOT / "data" / "benefitpilot.db"
    cache_dir: Path = REPO_ROOT / "data" / "cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
