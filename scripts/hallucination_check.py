#!/usr/bin/env python3
"""Identifier hallucination in answers: does each answer name a real snapshot entity?

For every answer, the target-type IDs are extracted the way the official scorer does
(ATT&CK for ATA, CWE for VCA) and checked against the CTIConnect snapshot:

  fabricated   the answer names an ID that does not exist in the snapshot
  no ID        the answer names no ID at all
  hedged       the answer names more than one ID (the scorer counts all of them)

No LLM calls; reads runs/<name>/predictions.jsonl.

    uv run python scripts/hallucination_check.py dev__closed_book__llama3-8b-fp16 \\
        dev__vanilla_rag__llama3-8b-fp16 dev__sgs__llama3-8b-fp16   # -> runs/hallucination__dev.json
"""

from __future__ import annotations

import argparse
import json

from ctiattr.config import RUNS
from ctiattr.snapshot import Snapshot
from ctiattr.vendor.cticonnect_metrics import extract_ids

KIND = {"ata": "mitre", "vca": "cwe"}


def check(preds: list[dict], valid: dict[str, set[str]]) -> dict:
    n = len(preds)
    no_id = fabricated = hedged = ids_total = ids_fake = 0
    examples = []
    for p in preds:
        kind = KIND[p["task"]]
        ids = extract_ids(p["prediction"], kind)
        fake = sorted(i for i in ids if i.upper() not in valid[kind])
        ids_total += len(ids)
        ids_fake += len(fake)
        no_id += not ids
        fabricated += bool(fake)
        hedged += len(ids) > 1
        if fake and len(examples) < 5:
            examples.append({"id": p["id"], "fabricated": fake, "gold": p["gold"]})
    return {"answers": n, "fabricated": round(fabricated / n, 3), "no_id": round(no_id / n, 3),
            "hedged": round(hedged / n, 3), "ids": ids_total, "fabricated_ids": ids_fake,
            "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="hallucination__dev.json")
    args = ap.parse_args()
    s = Snapshot()
    valid = {"mitre": s.technique_ids, "cwe": s.cwe_ids}
    out = {}
    for name in args.runs:
        preds = [json.loads(l) for l in (RUNS / name / "predictions.jsonl").open()]
        out[name] = {"all": check(preds, valid),
                     **{t: check([p for p in preds if p["task"] == t], valid) for t in ("ata", "vca")}}
        a = out[name]["all"]
        print(f"{name:40} fabricated {a['fabricated']:.0%} ({a['fabricated_ids']}/{a['ids']} IDs)  "
              f"no ID {a['no_id']:.0%}  hedged {a['hedged']:.0%}")
    (RUNS / args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
