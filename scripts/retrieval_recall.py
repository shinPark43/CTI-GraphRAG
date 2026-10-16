#!/usr/bin/env python3
"""Recall@k of the gold ID under CTIConnect's vanilla retrieval (no LLM).

    uv run python scripts/retrieval_recall.py --subset mini20
"""

from __future__ import annotations

import argparse
import json
import time

from ctiattr.config import RUNS, EmbedConfig
from ctiattr.data import TASK_KB, TASKS, load_all
from ctiattr.retrieval import Embedder, Retriever
from ctiattr.split import load_split

KS = (1, 5, 10, 20, 50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["dev", "mini20"], default="mini20")
    args = ap.parse_args()

    ids = set(load_split()[args.subset])
    qas = [qa for qa in load_all() if qa.id in ids]
    retriever = Retriever(Embedder(EmbedConfig.from_env()))
    t0 = time.time()
    for kind in sorted({TASK_KB[t] for t in TASKS}):
        n = len(retriever.index(kind).entries)
        print(f"index {kind}: {n} entries ({time.time() - t0:.0f}s)")

    rows = []
    for qa in qas:
        hits = retriever.retrieve(qa.question, TASK_KB[qa.task], k=max(KS))
        ranked = [h.doc_id.upper() for h in hits]
        rank = next((i + 1 for i, d in enumerate(ranked) if d in qa.gold), None)
        rows.append({"id": qa.id, "task": qa.task, "gold": qa.gold, "rank": rank,
                     "top5": ranked[:5]})

    print(f"\nRecall@k ({args.subset}, raw question as query)")
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        cells = "  ".join(f"@{k}={sum(1 for r in rs if r['rank'] and r['rank'] <= k) / len(rs):.2f}"
                          for k in KS)
        print(f"  {task} (n={len(rs)}): {cells}")
    out = RUNS / f"retrieval_recall__{args.subset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"per-question ranks -> {out}")


if __name__ == "__main__":
    main()
