#!/usr/bin/env python3
"""Audit the LLM report layer: sample GROUNDS edges per confidence tier for hand labels,
then score the labels.

    uv run python scripts/extraction_audit.py sample   # -> data/derived/extraction_audit.csv
    (fill the `correct` column: 1 = the ID is a right ATT&CK mapping for the behavior in
     its chunk, 0 = wrong technique or wrong sub-technique, or not an attacker behavior)
    uv run python scripts/extraction_audit.py score    # per-tier precision, overall precision

The overall precision weights each tier's precision by its share of all 3,512 edges.
Tier precision becomes the edge confidence written by scripts/load_graph.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random

from ctiattr.config import ROOT, EmbedConfig
from ctiattr.provenance import AUDIT, TIERS, audit_precision, extraction_signals, wilson
from ctiattr.retrieval import Embedder
from ctiattr.snapshot import Snapshot

DERIVED = ROOT / "data" / "derived"
PER_TIER = 25
FIELDS = ["tier", "chunk_id", "technique_id", "official_title", "model_name", "name_agreement", "sem_rank",
          "behavior", "chunk_text", "correct", "note"]


def load():
    s = Snapshot()
    ttps = [json.loads(l) for l in (DERIVED / "report_ttps.jsonl").open()]
    chunks = {c["chunk_id"]: c for c in map(json.loads, (DERIVED / "report_chunks.jsonl").open())}
    sig = extraction_signals(s, Embedder(EmbedConfig.from_env()), ttps, chunks)
    return s, ttps, chunks, sig


def sample() -> None:
    if AUDIT.exists():
        raise SystemExit(f"{AUDIT} exists; delete it to draw a new sample")
    s, ttps, chunks, sig = load()
    title = {e.doc_id: e.title for e in s.mitre}
    rng = random.Random(6338)
    rows = []
    for t in TIERS:
        idx = [i for i, x in enumerate(sig) if x.tier == t]
        for i in sorted(rng.sample(idx, min(PER_TIER, len(idx)))):
            b = ttps[i]
            rows.append({"tier": t, "chunk_id": b["chunk_id"], "technique_id": b["technique_id"],
                         "official_title": title[b["technique_id"]], "model_name": b["technique_name"],
                         "name_agreement": sig[i].name_agreement, "sem_rank": sig[i].sem_rank,
                         "behavior": b["behavior"], "chunk_text": chunks[b["chunk_id"]]["text"],
                         "correct": "", "note": ""})
    with AUDIT.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} edges -> {AUDIT}")


def score() -> None:
    _, _, _, sig = load()
    share = {t: sum(x.tier == t for x in sig) for t in TIERS}
    prec = audit_precision()
    total = sum(share.values())
    for t in TIERS:
        p = prec.get(t)
        print(f"  {t:10} {share[t]:5} edges ({share[t] / total:.1%})  "
              + (f"precision {p['precision']:.2f} [{p['ci95'][0]:.2f}, {p['ci95'][1]:.2f}] ({p['correct']}/{p['n']})"
                 if p else "not labelled"))
    if len(prec) == len(TIERS):
        overall = sum(prec[t]["precision"] * share[t] for t in TIERS) / total
        k = sum(p["correct"] for p in prec.values())
        n = sum(p["n"] for p in prec.values())
        print(f"  overall precision (tier-weighted) {overall:.3f}; unweighted sample {k}/{n} "
              f"[{wilson(k, n)[0]:.2f}, {wilson(k, n)[1]:.2f}]")
        out = {"tiers": prec, "edges": share, "overall_precision_weighted": round(overall, 3)}
        (ROOT / "runs" / "extraction_audit.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "score"])
    args = ap.parse_args()
    sample() if args.step == "sample" else score()


if __name__ == "__main__":
    main()
