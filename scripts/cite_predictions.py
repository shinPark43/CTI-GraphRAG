#!/usr/bin/env python3
"""Add graph citations to an existing SGS run and check every cited path in Neo4j.

No LLM calls: the chosen ID and its card are read from runs/<name>/predictions.jsonl,
the citations are rebuilt with ctiattr.citations, and each citation's Cypher query
must return exactly one path from the loaded graph. Also summarizes how much graph
support correct and wrong answers have, and the latency recorded in the run.

    uv run python scripts/cite_predictions.py dev__sgs__llama3-8b-fp16
    # -> runs/<name>/citations.jsonl, runs/<name>/citations_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from statistics import mean, median

from ctiattr.config import RUNS, EmbedConfig
from ctiattr.data import load_all
from ctiattr.graph_db import GraphDB
from ctiattr.retrieval import Embedder
from ctiattr.sgs import SGS


def latency(preds: list[dict]) -> dict:
    total = [p["seconds"] for p in preds]
    llm = [sum(c["latency_s"] for c in p.get("calls", [])) for p in preds]
    calls = [len(p.get("calls", [])) for p in preds]
    return {"total_mean": round(mean(total), 2), "total_median": round(median(total), 2),
            "total_max": round(max(total), 2), "llm_mean": round(mean(llm), 2),
            "llm_share": round(sum(llm) / max(sum(total), 1e-9), 3), "calls_mean": round(mean(calls), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name under runs/, an SGS run")
    ap.add_argument("--no-verify", action="store_true", help="skip the Neo4j path check")
    args = ap.parse_args()

    run = RUNS / args.run
    preds = [json.loads(l) for l in (run / "predictions.jsonl").open()]
    preds = [p for p in preds if not p.get("error") and p.get("chosen")]
    qas = {qa.id: qa for qa in load_all()}
    sgs = SGS(Embedder(EmbedConfig.from_env()))
    db = None if args.no_verify else GraphDB()

    rows, verified, checked = [], 0, 0
    support = defaultdict(list)  # correct / wrong -> citations per answer
    kinds = Counter()
    for p in preds:
        qa = qas[p["id"]]
        cites = sgs.citer.cite(qa, p["chosen"], p["cards"][p["chosen"]])
        ok = []
        for c in cites:
            kinds[c.kind] += 1
            if db is not None:
                n = db.run(c.cypher.replace("RETURN p", "RETURN count(p) AS n"))[0]["n"]
                ok.append(n == 1)
                checked += 1
                verified += n == 1
        correct = p["chosen"].upper() == qa.gold[0].upper()
        support["correct" if correct else "wrong"].append(
            len([c for c in cites if c.kind != "taxonomy"]))
        rows.append({"id": p["id"], "task": p["task"], "chosen": p["chosen"], "correct": correct,
                     "citations": [c.model_dump(exclude_none=True) for c in cites],
                     "verified": ok if db is not None else None})

    with (run / "citations.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {
        "run": args.run, "answers": len(rows),
        "with_evidence_path": sum(1 for r in rows if any(c["kind"] != "taxonomy" for c in r["citations"])),
        "citations": sum(kinds.values()), "by_kind": dict(kinds),
        "verified_in_neo4j": f"{verified}/{checked}" if db is not None else None,
        "evidence_paths_per_answer": {k: round(mean(v), 2) for k, v in support.items() if v},
        "correct_vs_wrong": {"correct": len(support["correct"]), "wrong": len(support["wrong"])},
        "latency_s": {t: latency([p for p in preds if p["task"] == t]) for t in ("ata", "vca")},
    }
    (run / "citations_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
