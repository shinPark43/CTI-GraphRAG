#!/usr/bin/env python3
"""Side-by-side table of finished runs.

    uv run python scripts/compare_runs.py mini20__vanilla_rag__llama3-8b-q4 mini20__vanilla_rag__llama3-8b-fp16
    uv run python scripts/compare_runs.py --glob 'mini20__*'

Columns: per-task and overall F1, mean IDs per answer, and how many questions
had the gold ID among the candidates shown to the model.
"""

from __future__ import annotations

import argparse
import json

from ctiattr.config import RUNS


def gold_shown(rec: dict) -> bool | None:
    if "retrieved" not in rec:
        return None
    return any(rec["gold"][0] in [d.upper() for d in ids] for ids in rec["retrieved"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", default=None)
    args = ap.parse_args()
    names = list(args.runs)
    if args.glob:
        names += sorted(p.name for p in RUNS.glob(args.glob) if (p / "scores.json").exists())

    print(f"{'run':48} {'ATA':>5} {'VCA':>5} {'all':>5} {'ids/q':>6} {'shown':>6} {'err':>4}")
    for name in names:
        s = json.loads((RUNS / name / "scores.json").read_text())
        recs = [json.loads(l) for l in (RUNS / name / "predictions.jsonl").open()]
        shown = [gold_shown(r) for r in recs]
        shown_s = "—" if shown[0] is None else f"{sum(shown)}/{len(shown)}"
        pt = s["per_task"]
        print(f"{name:48} {pt.get('ata', {}).get('f1', 0):5.2f} {pt.get('vca', {}).get('f1', 0):5.2f} "
              f"{s['overall']['f1']:5.2f} {s['mean_pred_ids']:6.2f} {shown_s:>6} {s.get('n_errors', 0):4}")


if __name__ == "__main__":
    main()
