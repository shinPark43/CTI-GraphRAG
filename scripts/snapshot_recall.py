#!/usr/bin/env python3
"""Candidate recall@k of snapshot-derived generators vs CTIConnect's retriever (no LLM).

    uv run python scripts/snapshot_recall.py --subset dev
"""

from __future__ import annotations

import argparse
import json

from ctiattr.candidates import CandidateGenerator
from ctiattr.config import RUNS, EmbedConfig
from ctiattr.data import TASK_KB, load_all
from ctiattr.retrieval import Embedder, Retriever
from ctiattr.split import load_split

KS = (1, 5, 10, 20)


def recall(rows, task, key):
    rs = [r for r in rows if r["task"] == task and key in r["ranks"]]
    return [sum(1 for r in rs if r["ranks"][key] and r["ranks"][key] <= k) / len(rs) for k in KS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["dev", "mini20"], default="dev")
    args = ap.parse_args()
    ids = set(load_split()[args.subset])
    qas = [qa for qa in load_all() if qa.id in ids]

    emb = Embedder(EmbedConfig.from_env())
    base = Retriever(emb)
    gen = CandidateGenerator(emb)

    rows = []
    for qa in qas:
        gold = qa.gold[0]
        rankings = {"baseline (question, raw KB)":
                    [h.doc_id.upper() for h in base.retrieve(qa.question, TASK_KB[qa.task], k=50)]}
        found = (gen.ata if qa.task == "ata" else gen.vca)(qa.question, qa.passage)
        rankings.update(found)
        ranks = {name: next((i + 1 for i, d in enumerate(r[:50]) if d.upper() == gold), None)
                 for name, r in rankings.items()}
        rows.append({"id": qa.id, "task": qa.task, "gold": gold, "ranks": ranks})

    for task in ("ata", "vca"):
        n = sum(r["task"] == task for r in rows)
        print(f"\n{task.upper()} (n={n})  " + "  ".join(f"@{k:<4}" for k in KS))
        for key in rows[[r["task"] for r in rows].index(task)]["ranks"]:
            print(f"  {key:30}" + "  ".join(f"{v:5.2f}" for v in recall(rows, task, key)))
    out = RUNS / f"snapshot_recall__{args.subset}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nper-question ranks -> {out}")


if __name__ == "__main__":
    main()
