#!/usr/bin/env python3
"""Stage 1 test of GDS outputs as retrieval signals: candidate recall@k on dev, no LLM.

Needs scripts/gds_analysis.py to have written pagerank and community onto the
nodes (corpus_kb graph only, so no report information leaks in). Rows rank all
ATT&CK techniques (ATA) or all CWEs (VCA):

  text               cos(q, x_v), text-embedding-3-large (what vanilla RAG retrieves)
  PR beta            cos + beta * pr_v, pr_v = log PageRank min-max scaled within the target kind
  community gamma    cos + gamma * w_c(v): the question's 10 nearest KB nodes by text vote
                     for their Louvain communities (weights max(cos, 0), normalized to sum 1)
  PR + community     both terms, at each one's best setting
  SGS, SGS + GDS     the frozen SGS fusion, and the same with the best GDS ranking added by RRF

As in graph_embed_recall.py, each row is scored for the question alone (q) and
for the RRF of question and passage (q+p). Also reports where the gold answers
sit in the graph (isolated, PageRank percentile, in an anchor community).

    uv run python scripts/gds_recall.py      # -> runs/gds_recall__dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_embed_recall import KS, rank_of, sgs_rankings  # noqa: E402

from ctiattr import gds  # noqa: E402
from ctiattr.candidates import rrf  # noqa: E402
from ctiattr.config import RUNS, EmbedConfig  # noqa: E402
from ctiattr.data import load_all  # noqa: E402
from ctiattr.graph_db import GraphDB  # noqa: E402
from ctiattr.graph_embed import GraphEmbedRetriever  # noqa: E402
from ctiattr.retrieval import Embedder  # noqa: E402
from ctiattr.sgs import SGS  # noqa: E402
from ctiattr.split import load_split  # noqa: E402

BETAS = (0.01, 0.02, 0.05, 0.1)
GAMMAS = (0.02, 0.05, 0.1, 0.2)
ANCHORS = 10


class GDSRanker:
    def __init__(self, ge: GraphEmbedRetriever, score: dict[str, dict]):
        self.ge = ge
        ids = ge.g.ids
        self.kb = np.array([k != "beh" for k in ge.g.kind])  # anchors come from corpus_kb nodes only
        self.comm = np.array([score.get(n, {}).get("community", -1) if self.kb[i] else -1
                              for i, n in enumerate(ids)])
        self.pr = {}
        for task in ("ata", "vca"):
            idx = ge.targets(task)
            lp = np.log(np.array([score[ids[i]]["pagerank"] for i in idx]))
            self.pr[task] = (lp - lp.min()) / max(lp.max() - lp.min(), 1e-12)

    def votes(self, v: np.ndarray) -> dict[int, float]:
        sims = np.where(self.kb, self.ge.X @ v, -np.inf)
        near = np.argpartition(-sims, ANCHORS)[:ANCHORS]
        w = defaultdict(float)
        for i in near:
            w[int(self.comm[i])] += max(float(sims[i]), 0.0)
        total = sum(w.values()) or 1.0
        return {c: x / total for c, x in w.items()}

    def rank(self, task: str, v: np.ndarray, beta: float = 0.0, gamma: float = 0.0) -> list[str]:
        idx = self.ge.targets(task)
        s = self.ge.X[idx] @ v
        if beta:
            s = s + beta * self.pr[task]
        if gamma:
            w = self.votes(v)
            s = s + gamma * np.array([w.get(int(self.comm[i]), 0.0) for i in idx])
        return [self.ge.g.ids[idx[j]] for j in np.argsort(-s)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["dev", "mini20"], default="dev")
    args = ap.parse_args()
    ids = set(load_split()[args.subset])
    qas = [qa for qa in load_all() if qa.id in ids]

    score = gds.scores(GraphDB())
    if any(s["pagerank"] is None for s in score.values()):
        raise SystemExit("pagerank/community missing on nodes: run scripts/gds_analysis.py first")
    emb = Embedder(EmbedConfig.from_env())
    ge = GraphEmbedRetriever(emb)
    r = GDSRanker(ge, score)
    qv = {qa.id: emb.embed_one(qa.question) for qa in qas}
    pv = {qa.id: emb.embed_one(qa.passage) for qa in qas}
    ranks: dict[str, dict[str, int | None]] = {qa.id: {} for qa in qas}
    lists: dict[str, dict[str, list[str]]] = {qa.id: {} for qa in qas}

    def record(name, qa, **kw):
        by_q, by_p = r.rank(qa.task, qv[qa.id], **kw), r.rank(qa.task, pv[qa.id], **kw)
        lists[qa.id][name] = rrf([by_q, by_p])
        ranks[qa.id][f"{name} | q"] = rank_of(by_q, qa.gold[0])
        ranks[qa.id][f"{name} | q+p"] = rank_of(lists[qa.id][name], qa.gold[0])

    def recall(rows, name):
        return [sum(1 for qa in rows if ranks[qa.id][name] and ranks[qa.id][name] <= k) / len(rows) for k in KS]

    for qa in qas:
        record("text", qa)
        for b in BETAS:
            record(f"PR {b}", qa, beta=b)
        for g in GAMMAS:
            record(f"community {g}", qa, gamma=g)

    sgs = SGS(emb)
    sgs_lists = {qa.id: sgs_rankings(sgs, qa) for qa in qas}
    out = {"subset": args.subset, "tasks": {}}
    for task in ("ata", "vca"):
        rows = [qa for qa in qas if qa.task == task]
        best = {}
        for v in ("PR", "community"):
            cand = [n.removesuffix(" | q+p") for n in ranks[rows[0].id] if n.startswith(v + " ") and n.endswith("q+p")]
            best[v] = max(cand, key=lambda n: (recall(rows, f"{n} | q+p")[1], recall(rows, f"{n} | q+p")[3]))
        b, g = float(best["PR"].split()[1]), float(best["community"].split()[1])
        for qa in rows:
            record("PR + community", qa, beta=b, gamma=g)
            ranks[qa.id]["SGS (frozen)"] = rank_of(sgs_lists[qa.id], qa.gold[0])
            for v in ("PR", "community", "PR + community"):
                name = best.get(v, v)
                ranks[qa.id][f"SGS + {name}"] = rank_of(rrf([sgs_lists[qa.id], lists[qa.id][name]]), qa.gold[0])
        table = {n: recall(rows, n) for n in ranks[rows[0].id]}

        # where the gold answers sit in the graph
        tgt = [ge.g.ids[i] for i in ge.targets(task)]
        prs = np.array([score[t]["pagerank"] for t in tgt])
        gold = [qa.gold[0] for qa in rows if qa.gold[0] in score]
        pct = [float((prs < score[t]["pagerank"]).mean()) for t in gold]
        in_anchor = [int(score[qa.gold[0]]["community"]) in r.votes(qv[qa.id]) for qa in rows if qa.gold[0] in score]
        diag = {"gold_in_graph": len(gold), "gold_isolated": sum(1 for t in gold if score[t]["degree"] == 0),
                "gold_pagerank_percentile_median": round(float(np.median(pct)), 3),
                "gold_in_anchor_community": round(sum(in_anchor) / max(len(in_anchor), 1), 3),
                "targets_isolated": round(float(np.mean([score[t]["degree"] == 0 for t in tgt])), 3)}
        out["tasks"][task] = {"n": len(rows), "best": best, "recall": table, "gold": diag}

        print(f"\n{task.upper()} (n={len(rows)})" + " " * 30 + "  ".join(f"@{k:<4}" for k in KS))
        for n in table:
            if n.endswith("| q+p") or n.startswith("SGS") or n == "text | q":
                print(f"  {n:40}" + "  ".join(f"{x:5.2f}" for x in table[n]))
        print(f"  gold: {diag}")

    path = RUNS / f"gds_recall__{args.subset}.json"
    path.write_text(json.dumps({**out, "ranks": ranks}, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
