#!/usr/bin/env python3
"""Stage 1 of the graph-embedding test: candidate recall@k on dev, no LLM calls.

Rows (each ranks all ATT&CK techniques for ATA, all CWEs for VCA):
  text                  text-embedding-3-large alone (what vanilla RAG retrieves)
  A edges/alpha/K       propagated text embeddings (graph_embed.propagate)
  B q/lambda            text + node2vec, question anchored on its 10 nearest KB nodes
  SGS, SGS + GE         the frozen SGS fusion, and the same with a GE ranking added to the RRF
Each row is scored for two queries: the question alone (q, as vanilla RAG) and the
RRF of question and passage (q+p, as SGS). Report edges from the question's own
report are always removed; node2vec is trained on the corpus_kb graph only.

    uv run python scripts/graph_embed_recall.py
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from ctiattr.candidates import rrf
from ctiattr.config import RUNS, EmbedConfig
from ctiattr.data import load_all
from ctiattr.graph_embed import GraphEmbedRetriever
from ctiattr.retrieval import Embedder
from ctiattr.sgs import SGS
from ctiattr.split import load_split

KS = (1, 5, 10, 20)
EDGE_SETS = {"taxonomy": ("hier", "tactic"),
             "links": ("capec", "cve"),
             "kb": ("hier", "tactic", "capec", "cve"),
             "report": ("report",),
             "kb+report": ("hier", "tactic", "capec", "cve", "report")}
ALPHAS = (0.2, 0.4, 0.6)
HOPS = (1, 2)
N2V_Q = (0.5, 1.0, 2.0)
LAMBDAS = (0.1, 0.25, 0.5, 1.0)


def rank_of(ranking, gold):
    return next((i + 1 for i, d in enumerate(ranking[:50]) if d.upper() == gold), None)


def sgs_rankings(sgs: SGS, qa) -> list[str]:
    """Full SGS fused ranking (SGS.ata_candidates / vca_candidates keep only its top 5)."""
    if qa.task == "vca":
        return sgs.gen.vca(qa.question, qa.passage)["fused"]
    pv = sgs.emb.embed_one(qa.passage)
    sims = sgs.behaviors @ pv
    nearest = [i for i in np.argsort(-sims) if sgs.ttps[i]["report_id"] != qa.report_id][:30]
    score = defaultdict(float)
    for i in nearest:
        score[sgs.ttps[i]["technique_id"]] += float(sims[i])
    vote = sorted(score, key=lambda t: -score[t])
    base = rrf([[t for t, _ in sgs.all_tech.rank(qa.question)], [t for t, _ in sgs.all_tech.rank(qa.passage)]])
    return rrf([base, vote])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["dev", "mini20"], default="dev")
    args = ap.parse_args()
    ids = set(load_split()[args.subset])
    qas = sorted((qa for qa in load_all() if qa.id in ids), key=lambda q: q.report_id)

    emb = Embedder(EmbedConfig.from_env())
    ge = GraphEmbedRetriever(emb)
    print(f"graph: {len(ge.g.ids)} nodes, edges {ge.edge_counts()}")
    qv = {qa.id: emb.embed_one(qa.question) for qa in qas}
    pv = {qa.id: emb.embed_one(qa.passage) for qa in qas}
    ranks: dict[str, dict[str, int | None]] = {qa.id: {} for qa in qas}
    lists: dict[str, dict[str, list[str]]] = {qa.id: {} for qa in qas}  # kept for the SGS + GE rows

    def record(name, qa, by_q, by_p):
        lists[qa.id][name] = rrf([by_q, by_p])
        ranks[qa.id][f"{name} | q"] = rank_of(by_q, qa.gold[0])
        ranks[qa.id][f"{name} | q+p"] = rank_of(lists[qa.id][name], qa.gold[0])

    for qa in qas:
        idx = ge.targets(qa.task)
        t = lambda v: [ge.g.ids[idx[j]] for j in np.argsort(-(ge.X[idx] @ v))]
        record("text", qa, t(qv[qa.id]), t(pv[qa.id]))

    t0 = time.time()
    for es, types in EDGE_SETS.items():
        for a in ALPHAS:
            for k in HOPS:
                for qa in qas:  # sorted by report, so each held-out matrix is built once
                    r = lambda v: ge.rank_propagated(qa.task, v, types, a, k, qa.report_id)
                    record(f"A {es} a={a} K={k}", qa, r(qv[qa.id]), r(pv[qa.id]))
        print(f"  A {es} done ({time.time() - t0:.0f}s)", flush=True)

    kb = EDGE_SETS["kb"]
    for q in N2V_Q:
        G = ge.structural(kb, q=q)
        for lam in LAMBDAS:
            for qa in qas:
                r = lambda v: ge.rank_node2vec(qa.task, v, G, lam)
                record(f"B q={q} lam={lam}", qa, r(qv[qa.id]), r(pv[qa.id]))
        print(f"  B q={q} done ({time.time() - t0:.0f}s)", flush=True)

    sgs = SGS(emb)
    sgs_lists = {qa.id: sgs_rankings(sgs, qa) for qa in qas}
    for qa in qas:
        ranks[qa.id]["SGS (frozen)"] = rank_of(sgs_lists[qa.id], qa.gold[0])

    out = {"subset": args.subset, "edges": ge.edge_counts(), "nodes": len(ge.g.ids), "tasks": {}}
    for task in ("ata", "vca"):
        rows = [qa for qa in qas if qa.task == task]
        names = list(ranks[rows[0].id])
        table = {n: [sum(1 for qa in rows if ranks[qa.id][n] and ranks[qa.id][n] <= k) / len(rows) for k in KS]
                 for n in names}
        # best of each variant by recall@5 (ties -> @20), for the q+p query
        best = {}
        for v in ("A", "B"):
            cand = [n for n in names if n.startswith(v + " ") and n.endswith("| q+p")]
            best[v] = max(cand, key=lambda n: (table[n][1], table[n][3]))
        for v, n in best.items():
            name = n.removesuffix(" | q+p")
            key = f"SGS + GE ({name})"
            for qa in rows:
                ranks[qa.id][key] = rank_of(rrf([sgs_lists[qa.id], lists[qa.id][name]]), qa.gold[0])
            table[key] = [sum(1 for qa in rows if ranks[qa.id][key] and ranks[qa.id][key] <= k) / len(rows)
                          for k in KS]
        out["tasks"][task] = {"n": len(rows), "recall": table, "best": best}

        print(f"\n{task.upper()} (n={len(rows)})" + " " * 34 + "  ".join(f"@{k:<4}" for k in KS))
        top = sorted((n for n in table if n.startswith(("A ", "B "))), key=lambda n: (-table[n][1], -table[n][3]))
        show = ["text | q", "text | q+p", *top[:8], "SGS (frozen)", *[n for n in table if n.startswith("SGS +")]]
        for n in show:
            print(f"  {n:44}" + "  ".join(f"{v:5.2f}" for v in table[n]))

    path = RUNS / f"graph_embed_recall__{args.subset}.json"
    path.write_text(json.dumps({**out, "ranks": ranks}, indent=1))
    print(f"\nall {len(ranks[qas[0].id])} rows and per-question ranks -> {path}")


if __name__ == "__main__":
    main()
