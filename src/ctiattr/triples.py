"""Score predictions as (report, predicate, ID) triples against the gold triples.

A run's answers become triples: each question contributes
(its source report, DESCRIBES_TECHNIQUE | DESCRIBES_WEAKNESS, each predicted ID),
with IDs extracted and normalized by CTIConnect's official scorer functions.
Precision, recall and F1 are computed over the two triple sets, overall and
per predicate. Near misses (parent or child technique of a gold triple), IDs
absent from the snapshot and retired IDs are counted separately, never as hits.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ctiattr.vendor.cticonnect_metrics import extract_ids

PREDICATE = {"ata": "DESCRIBES_TECHNIQUE", "vca": "DESCRIBES_WEAKNESS"}
KIND = {"ata": "mitre", "vca": "cwe"}

Triple = tuple[str, str, str]


def load_gold(path: Path) -> tuple[set[Triple], set[str]]:
    data = json.loads(path.read_text())
    gold = {(t["subject"]["id"], t["predicate"], t["object"]["id"].upper()) for t in data["triples"]}
    questions = {t["source_question"] for t in data["triples"]}
    return gold, questions


def predicted_triples(predictions: list[dict], reports: dict[str, str]) -> set[Triple]:
    """predictions: run records with id, task, prediction; reports: question id -> report id."""
    out = set()
    for r in predictions:
        if r["id"] not in reports:
            continue
        for i in extract_ids(r.get("prediction") or "", KIND[r["task"]]):
            out.add((reports[r["id"]], PREDICATE[r["task"]], i.upper()))
    return out


def prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "predicted": len(pred), "gold": len(gold),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def score_triples(pred: set[Triple], gold: set[Triple], *, known_ids: set[str] | None = None,
                  retired: set[str] | None = None) -> dict:
    result = {"overall": prf(pred, gold), "per_predicate": {}}
    for p in sorted({t[1] for t in gold | pred}):
        result["per_predicate"][p] = prf({t for t in pred if t[1] == p}, {t for t in gold if t[1] == p})

    family = lambda i: i.split(".")[0]
    gold_by_report = defaultdict(set)
    for s, p, o in gold:
        gold_by_report[(s, p)].add(o)
    wrong = pred - gold
    result["diagnostics"] = {
        "near_miss_parent_or_child": sum(
            1 for s, p, o in wrong if p == PREDICATE["ata"]
            and any(family(o) == family(g) for g in gold_by_report[(s, p)])),
        "unknown_id": sum(1 for _, _, o in pred if known_ids is not None and o not in known_ids),
        "retired_id": sum(1 for _, _, o in pred if retired and o in retired),
    }
    return result
