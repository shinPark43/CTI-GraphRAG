"""Candidate generation from the CTIConnect snapshot.

Each generator returns a ranking (list of IDs, best first) for one passage;
`rrf` fuses rankings. Generators:

- dense / BM25 over the target taxonomy's own records (title + contents)
- CAPEC bridge: passage -> similar CAPEC patterns -> their CWE / ATT&CK links
- CVE examples (VCA): passage -> similar CVE descriptions -> their CWEs
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from ctiattr.retrieval import Embedder
from ctiattr.snapshot import Snapshot

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was "
            "were which with can may their they these those into such not".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class DenseIndex:
    def __init__(self, ids: list[str], texts: list[str], embedder: Embedder):
        self.ids = ids
        self.embedder = embedder
        self.matrix = embedder.embed(texts)

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.embed_one(query)

    def rank(self, query: str) -> list[tuple[str, float]]:
        s = self.scores(query)
        order = np.argsort(-s)
        return [(self.ids[i], float(s[i])) for i in order]


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.ids = ids
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query: str) -> list[tuple[str, float]]:
        q = set(tokenize(query))
        out = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                out.append((self.ids[i], s))
        return sorted(out, key=lambda x: -x[1])


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, doc in enumerate(ranking, 1):
            score[doc] += 1.0 / (k + r)
    return sorted(score, key=lambda d: -score[d])


def _aggregate(hits: list[tuple[str, float]], links, top: int) -> list[str]:
    """Score each linked target by the similarity of the records that point to it."""
    score: dict[str, float] = defaultdict(float)
    for rid, sim in hits[:top]:
        for target in links(rid):
            score[target] += sim
    return sorted(score, key=lambda t: -score[t])


class CandidateGenerator:
    def __init__(self, embedder: Embedder, snapshot: Snapshot | None = None):
        self.s = snapshot or Snapshot()
        s = self.s
        tech = [e for e in s.mitre if e.doc_id in s.answerable_techniques]
        self.tech_ids = [e.doc_id for e in tech]
        tech_text = [f"{e.title}. {e.text}" for e in tech]
        self.tech_dense = DenseIndex(self.tech_ids, tech_text, embedder)
        self.tech_bm25 = BM25(self.tech_ids, tech_text)

        self.cwe_ids = [e.doc_id.upper() for e in s.cwe]
        cwe_text = [f"{e.title}. {e.text}" for e in s.cwe]
        self.cwe_dense = DenseIndex(self.cwe_ids, cwe_text, embedder)
        self.cwe_bm25 = BM25(self.cwe_ids, cwe_text)

        self.capec = {c.id: c for c in s.capec}
        self.capec_dense = DenseIndex(list(self.capec), [f"{c.name}. {c.description}" for c in s.capec],
                                      embedder)
        self.cve = {c.id: c for c in s.cve}
        self.cve_dense = DenseIndex(list(self.cve), [c.description for c in s.cve], embedder)

    # --- ATA ---------------------------------------------------------------

    # `fused` is the configuration chosen on the dev split (scripts/snapshot_recall.py);
    # the other rankings are kept for ablations.

    def ata(self, question: str, passage: str) -> dict[str, list[str]]:
        answerable = self.s.answerable_techniques
        capec_hits = self.capec_dense.rank(passage)
        bridge = [t for t in _aggregate(capec_hits, lambda c: self.capec[c].techniques, top=10)
                  if t in answerable]
        out = {
            "dense_question": [t for t, _ in self.tech_dense.rank(question)],
            "dense_passage": [t for t, _ in self.tech_dense.rank(passage)],
            "bm25": [t for t, _ in self.tech_bm25.rank(passage)],
            "capec": bridge,
        }
        out["fused"] = rrf([out["dense_question"], out["dense_passage"]])
        return out

    # --- VCA ---------------------------------------------------------------

    def vca(self, question: str, passage: str) -> dict[str, list[str]]:
        capec_hits = self.capec_dense.rank(passage)
        cve_hits = self.cve_dense.rank(passage)
        out = {
            "dense_question": [w for w, _ in self.cwe_dense.rank(question)],
            "dense_passage": [w for w, _ in self.cwe_dense.rank(passage)],
            "bm25": [w for w, _ in self.cwe_bm25.rank(passage)],
            "capec": _aggregate(capec_hits, lambda c: self.capec[c].cwes, top=10),
            "cve": _aggregate(cve_hits, lambda c: self.cve[c].cwes, top=20),
        }
        out["fused"] = rrf([out["dense_question"], out["capec"], out["cve"]])
        return out
