#!/usr/bin/env python3
"""Triple-level P/R/F1 of a run against data/gold_triples.json (69 dev triples).

    uv run python scripts/verify_triples.py dev__sgs__llama3-8b-fp16 dev__vanilla_rag__llama3-8b-fp16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctiattr.config import ROOT, RUNS
from ctiattr.data import load_all
from ctiattr.snapshot import Snapshot
from ctiattr.triples import load_gold, predicted_triples, score_triples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gold", default=str(ROOT / "data" / "gold_triples.json"))
    args = ap.parse_args()

    gold, questions = load_gold(Path(args.gold))
    reports = {qa.id: qa.report_id for qa in load_all() if qa.id in questions}
    s = Snapshot()
    known = s.technique_ids | s.cwe_ids

    print(f"gold: {len(gold)} triples from {len(questions)} questions")
    print(f"{'run':40} {'P':>6} {'R':>6} {'F1':>6}   TECHNIQUE F1  WEAKNESS F1   near-miss unknown retired")
    for name in args.runs:
        preds = [json.loads(l) for l in (RUNS / name / "predictions.jsonl").open()]
        res = score_triples(predicted_triples(preds, reports), gold, known_ids=known, retired=set(s.retired))
        o, pp, d = res["overall"], res["per_predicate"], res["diagnostics"]
        print(f"{name:40} {o['precision']:6.3f} {o['recall']:6.3f} {o['f1']:6.3f}   "
              f"{pp.get('DESCRIBES_TECHNIQUE', {}).get('f1', 0):12.3f} {pp.get('DESCRIBES_WEAKNESS', {}).get('f1', 0):12.3f}"
              f"   {d['near_miss_parent_or_child']:9} {d['unknown_id']:7} {d['retired_id']:7}")
        (RUNS / name / "triple_scores.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
