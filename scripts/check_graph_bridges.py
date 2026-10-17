#!/usr/bin/env python3
"""Check that SGS candidates from Neo4j (Cypher + vector indexes) match the
in-memory implementation on the dev split. No LLM calls.

    uv run python scripts/check_graph_bridges.py
"""

from __future__ import annotations

from ctiattr.config import EmbedConfig
from ctiattr.data import load_all
from ctiattr.graph_db import GraphDB
from ctiattr.retrieval import Embedder
from ctiattr.sgs import SGS
from ctiattr.split import load_split


def main() -> None:
    emb = Embedder(EmbedConfig.from_env())
    db = GraphDB()
    mem, neo = SGS(emb), SGS(emb, graph=db)
    neo.s, neo.gen = mem.s, mem.gen  # share the built in-memory structures; only candidates differ
    dev = set(load_split()["dev"])
    qs = [q for q in load_all() if q.id in dev]
    same_list = same_set = gold_same = 0
    diffs = []
    for q in qs:
        get = "ata_candidates" if q.task == "ata" else "vca_candidates"
        a, _ = getattr(mem, get)(q)
        b, _ = getattr(neo, get)(q)
        same_list += a == b
        same_set += set(a) == set(b)
        gold_same += (q.gold[0] in a) == (q.gold[0] in b)
        if a != b:
            diffs.append((q.id, a, b))
    n = len(qs)
    print(f"dev questions: {n}")
    print(f"  identical top-5 lists (same order): {same_list}/{n}")
    print(f"  identical top-5 sets:               {same_set}/{n}")
    print(f"  same gold-in-top-5 outcome:         {gold_same}/{n}")
    for d in diffs[:5]:
        print("  differs:", d)
    db.close()


if __name__ == "__main__":
    main()
