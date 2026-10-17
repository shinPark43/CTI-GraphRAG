"""Graph embeddings over the CTIConnect snapshot graph, for retrieval (Section 5).

The graph is the one loaded into Neo4j (schema.cypher), built from corpus_kb and
the report layer only:

    hier    technique - parent technique (SUBTECHNIQUE_OF), retired ID - replacement (REPLACED_BY)
    tactic  technique - tactic (IN_TACTIC)
    capec   CAPEC pattern - CWE, CAPEC pattern - technique
    cve     CVE - CWE (HAS_WEAKNESS)
    report  report behavior - technique (GROUNDS); never from the question's own report

Two ways to combine a text embedding with the graph, both scored against the
question's text-embedding-3-large vector:

A. Propagated text embeddings (training-free GraphSAGE-mean / APPNP form):
       H(0) = X,   H(k+1) = (1 - alpha) X + alpha P H(k),   P = D^-1 A
   so each node's vector mixes in its K-hop neighborhood and stays in text space.
B. Text + node2vec: a structural embedding g_v learned from biased random walks
   (skip-gram, negative sampling). The question has no node, so it is anchored to
   the graph through its nearest KB nodes by text:
       g_q = normalize(sum_i s_i g_i) over the 10 nearest nodes, s_i = max(cos, 0)
       score(v) = cos(q, x_v) + lambda cos(g_q, g_v)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ctiattr.config import ROOT
from ctiattr.retrieval import Embedder
from ctiattr.snapshot import Snapshot

EDGE_TYPES = ("hier", "tactic", "capec", "cve", "report")


@dataclass
class SnapshotGraph:
    ids: list[str]                      # node ids: T1059, tactic:execution, CWE-79, CAPEC-66, CVE-..., beh:12
    kind: list[str]                     # tech | tactic | cwe | capec | cve | beh
    texts: list[str]                    # text embedded for each node
    edges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    report_of: dict[int, str] = field(default_factory=dict)   # behavior node -> its report id

    @property
    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.ids)}

    def of_kind(self, kind: str) -> list[int]:
        return [i for i, k in enumerate(self.kind) if k == kind]


def build_graph(s: Snapshot | None = None) -> SnapshotGraph:
    s = s or Snapshot()
    g = SnapshotGraph([], [], [])

    def add(nid, kind, text):
        g.ids.append(nid), g.kind.append(kind), g.texts.append(text)
        return len(g.ids) - 1

    # Texts match the ones already embedded elsewhere (SGS, CandidateGenerator), so they hit the cache.
    tech = {e.doc_id: add(e.doc_id, "tech", f"{e.title}. {e.text}") for e in s.mitre}
    tactic = {}
    for t in tech:
        for name in s.tactics(t):
            if name not in tactic:
                tactic[name] = add(f"tactic:{name}", "tactic", f"ATT&CK tactic: {name.replace('-', ' ')}")
    cwe = {e.doc_id.upper(): add(e.doc_id.upper(), "cwe", f"{e.title}. {e.text}") for e in s.cwe}
    capec = {c.id: add(c.id, "capec", f"{c.name}. {c.description}") for c in s.capec}
    cve = {c.id: add(c.id, "cve", c.description) for c in s.cve}

    e = {k: [] for k in EDGE_TYPES}
    for t, i in tech.items():
        if s.parent(t) in tech:
            e["hier"].append((i, tech[s.parent(t)]))
        e["tactic"] += [(i, tactic[name]) for name in s.tactics(t)]
    e["hier"] += [(tech[old], tech[new]) for old, new in s.retired.items()]
    for c in s.capec:
        e["capec"] += [(capec[c.id], cwe[w]) for w in c.cwes]
        e["capec"] += [(capec[c.id], tech[t]) for t in c.techniques]
    for c in s.cve:
        e["cve"] += [(cve[c.id], cwe[w]) for w in c.cwes]

    for line in (ROOT / "data" / "derived" / "report_ttps.jsonl").open():
        b = json.loads(line)
        i = add(f"beh:{len(g.report_of)}", "beh", b["behavior"])
        g.report_of[i] = b["report_id"]
        e["report"].append((i, tech[b["technique_id"]]))
    g.edges = e
    return g


def adjacency(g: SnapshotGraph, types, exclude_report: str | None = None) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency over the chosen edge types, minus one report's behaviors."""
    rows, cols = [], []
    for t in types:
        for a, b in g.edges[t]:
            if exclude_report and (g.report_of.get(a) == exclude_report or g.report_of.get(b) == exclude_report):
                continue
            rows += [a, b]
            cols += [b, a]
    n = len(g.ids)
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0  # collapse duplicate edges
    return A


def propagate(X: np.ndarray, A: sp.csr_matrix, alpha: float, hops: int) -> np.ndarray:
    """Variant A: H(k+1) = (1-alpha) X + alpha D^-1 A H(k); isolated nodes keep their own vector."""
    deg = np.asarray(A.sum(1)).ravel()
    P = sp.diags(np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0).astype(np.float32)) @ A
    isolated = (deg == 0)[:, None]
    H = X
    for _ in range(hops):
        H = (1 - alpha) * X + alpha * np.where(isolated, H, P @ H)
    return H / np.linalg.norm(H, axis=1, keepdims=True).clip(1e-12)


def node2vec(A: sp.csr_matrix, *, dim: int = 128, walks: int = 10, length: int = 40, p: float = 1.0,
             q: float = 1.0, window: int = 5, seed: int = 13) -> np.ndarray:
    """Variant B: node2vec (Grover & Leskovec 2016) with rejection-sampled second-order walks.

    Step weight from prev -> cur -> x: 1/p if x == prev, 1 if x is a neighbor of prev, 1/q otherwise.
    Nodes with no edges get a zero vector.
    """
    from gensim.models import Word2Vec

    rng = random.Random(seed)
    nbrs = [A.indices[A.indptr[i]:A.indptr[i + 1]].tolist() for i in range(A.shape[0])]
    nset = [set(x) for x in nbrs]
    top = max(1.0, 1 / p, 1 / q)
    corpus = []
    nodes = [i for i in range(A.shape[0]) if nbrs[i]]
    for _ in range(walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start, rng.choice(nbrs[start])]
            while len(walk) < length:
                prev, cur = walk[-2], walk[-1]
                while True:
                    x = rng.choice(nbrs[cur])
                    w = 1 / p if x == prev else (1.0 if x in nset[prev] else 1 / q)
                    if rng.random() * top < w:
                        break
                walk.append(x)
            corpus.append([str(v) for v in walk])
    model = Word2Vec(corpus, vector_size=dim, window=window, min_count=0, sg=1, negative=5,
                     workers=4, epochs=1, seed=seed)
    out = np.zeros((A.shape[0], dim), np.float32)
    for i in nodes:
        out[i] = model.wv[str(i)]
    return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)


class GraphEmbedRetriever:
    """Rank one target kind (tech for ATA, cwe for VCA) by text + graph embeddings."""

    def __init__(self, embedder: Embedder, graph: SnapshotGraph | None = None):
        self.emb = embedder
        self.g = graph or build_graph()
        self.X = embedder.embed(self.g.texts)
        self._prop: dict[tuple, np.ndarray] = {}
        self._n2v: dict[tuple, np.ndarray] = {}

    def targets(self, task: str) -> list[int]:
        return self.g.of_kind("tech" if task == "ata" else "cwe")

    def propagated(self, types: tuple[str, ...], alpha: float, hops: int, report: str | None) -> np.ndarray:
        """Variant A vectors; recomputed per report only when report edges are used."""
        key = (types, alpha, hops, report if "report" in types else None)
        if key not in self._prop:
            if key[3] is not None:  # one held-out-report matrix at a time (each is ~150 MB)
                for k in [k for k in self._prop if k[3] is not None]:
                    del self._prop[k]
            self._prop[key] =propagate(self.X, adjacency(self.g, types, key[3]), alpha, hops)
        return self._prop[key]

    def rank_propagated(self, task, qv, types, alpha, hops, report) -> list[str]:
        H = self.propagated(tuple(types), alpha, hops, report)
        idx = self.targets(task)
        s = H[idx] @ qv
        return [self.g.ids[idx[j]] for j in np.argsort(-s)]

    def structural(self, types: tuple[str, ...], **kw) -> np.ndarray:
        key = (types, tuple(sorted(kw.items())))
        if key not in self._n2v:
            self._n2v[key] = node2vec(adjacency(self.g, types), **kw)
        return self._n2v[key]

    def rank_node2vec(self, task, qv, G: np.ndarray, lam: float, anchors: int = 10) -> list[str]:
        sims = self.X @ qv
        sims[np.linalg.norm(G, axis=1) == 0] = -np.inf  # anchor only on nodes inside the walk graph
        near = np.argpartition(-sims, anchors)[:anchors]
        gq = (np.maximum(sims[near], 0)[:, None] * G[near]).sum(0)
        gq /= max(np.linalg.norm(gq), 1e-12)
        idx = self.targets(task)
        s = self.X[idx] @ qv + lam * (G[idx] @ gq)
        return [self.g.ids[idx[j]] for j in np.argsort(-s)]

    def edge_counts(self) -> dict[str, int]:
        return {t: len(v) for t, v in self.g.edges.items()}
