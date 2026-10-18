"""Paths and endpoint settings.

The answering LLM and the embedder are configured separately: the LLM is any
OpenAI-compatible server (the local Ollama), embeddings always come from
OpenAI so retrieval matches the CTIConnect paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "cticonnect"
SPLITS = ROOT / "data" / "splits"
CACHE = ROOT / ".cache"
RUNS = ROOT / "runs"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls, model: str | None = None) -> "LLMConfig":
        base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        if not urlparse(base).path:  # bare host:port -> the OpenAI-compatible routes under /v1
            base += "/v1"
        return cls(
            base_url=base,
            api_key=os.environ.get("LLM_API_KEY", "ollama"),
            model=model or os.environ.get("LLM_MODEL", "llama3:8b"),
        )


@dataclass(frozen=True)
class EmbedConfig:
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "EmbedConfig":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (put it in .env)")
        return cls(api_key=key, model=os.environ.get("EMBED_MODEL", "text-embedding-3-large"))
