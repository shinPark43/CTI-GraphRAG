"""Dense retrieval over one knowledge base, ported from CTIConnect's
baselines/_shared (embedder.py, vector_index.py, kb.py, retriever.py; MIT).

Behaviour is kept identical to the paper's baselines: each entry is embedded as
"{title}. {contents}" with text-embedding-3-large, vectors are L2-normalized,
and search is exact cosine top-k. Embeddings are cached on disk by
sha1(model + text), so the corpus is embedded once.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from ctiattr.config import CACHE, EmbedConfig
from ctiattr.data import KBEntry, load_kb


class Embedder:
    def __init__(self, cfg: EmbedConfig, *, cache_dir: Path = CACHE / "emb", batch_size: int = 64):
        self.model = cfg.model
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(api_key=cfg.api_key)

    def _path(self, text: str) -> Path:
        return self.cache_dir / (hashlib.sha1(f"{self.model}\x00{text}".encode()).hexdigest() + ".npy")

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            p = self._path(t)
            if p.exists():
                out[i] = np.load(p)
            else:
                todo.append(i)
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start:start + self.batch_size]
            vecs = self._embed_batch([texts[i] or " " for i in idx])
            for j, i in enumerate(idx):
                out[i] = vecs[j]
                np.save(self._path(texts[i]), vecs[j])
        mat = np.vstack(out).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def _embed_batch(self, texts: list[str], retries: int = 5) -> np.ndarray:
        for attempt in range(retries):
            try:
                resp = self._client.embeddings.create(model=self.model, input=texts)
                return np.array([d.embedding for d in resp.data], dtype=np.float32)
            except Exception:  # noqa: BLE001
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    title: str
    text: str


class KBIndex:
    """Exact cosine search over one knowledge base."""

    def __init__(self, entries: list[KBEntry], matrix: np.ndarray):
        self.entries = entries
        self.matrix = matrix

    @classmethod
    def build(cls, kind: str, embedder: Embedder) -> "KBIndex":
        entries = load_kb(kind)
        texts = [f"{e.title}. {e.text}".strip() for e in entries]
        return cls(entries, embedder.embed(texts))

    def search(self, qv: np.ndarray, k: int = 5) -> list[Hit]:
        scores = self.matrix @ qv.astype(np.float32).reshape(-1)
        k = min(k, len(self.entries))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(self.entries[i].doc_id, float(scores[i]), self.entries[i].title,
                    self.entries[i].text) for i in top]


class Retriever:
    """One embedder + lazily built per-KB indexes."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._indexes: dict[str, KBIndex] = {}

    def index(self, kind: str) -> KBIndex:
        if kind not in self._indexes:
            self._indexes[kind] = KBIndex.build(kind, self.embedder)
        return self._indexes[kind]

    def retrieve(self, query: str, kind: str, k: int = 5) -> list[Hit]:
        return self.index(kind).search(self.embedder.embed_one(query), k=k)


def format_candidates(hits: list[Hit], *, max_chars: int = 600) -> str:
    """Render hits for the prompt exactly as CTIConnect's baselines do."""
    return "\n\n".join(f"[{h.doc_id}] {h.title}\n{h.text[:max_chars]}" for h in hits)
