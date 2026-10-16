# Vendored from CTIConnect (https://github.com/peng-gao-lab/CTIConnect) @ 554797d,
# evaluation/metrics.py — MIT License, Copyright (c) 2026 CTIConnect authors.
# Unmodified below this header; it is the official scorer for ATA/VCA.
"""Identifier-normalized Precision / Recall / F1 for Entity Linking and
Entity Attribution.

Both task families are graded by extracting the set of taxonomy identifiers
(CVE / CWE / CAPEC / ATT&CK) of the *target type* from the model's free-form
prediction, normalizing them to a canonical surface, and comparing against the
gold identifier set:

    Precision = |pred ∩ gold| / |pred|
    Recall    = |pred ∩ gold| / |gold|
    F1        = harmonic mean

Entity Linking (`single_id_match`) gold is a singleton; Entity Attribution
(`id_set_match`) gold is a set. The computation is identical — only the gold
field differs (`target_id` vs `target_ids`).

Multi-Doc Synthesis (`judge`) is NOT handled here — it requires the LLM judge
in `evaluation/judge/`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical identifier patterns. Each captures the full identifier token.
ID_PATTERNS: dict[str, re.Pattern] = {
    "cve":   re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE),
    "cwe":   re.compile(r"CWE-\d+", re.IGNORECASE),
    "capec": re.compile(r"CAPEC-\d+", re.IGNORECASE),
    # MITRE technique: T1059 or T1059.001 (sub-technique). Word-boundary so we
    # don't grab the 'T' inside other tokens.
    "mitre": re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE),
}


def normalize_id(raw: str, kind: str) -> str:
    """Canonicalize an identifier surface to uppercase, hyphenated form.

    >>> normalize_id("cwe 79", "cwe")
    'CWE-79'
    >>> normalize_id("t1059.001", "mitre")
    'T1059.001'
    >>> normalize_id("CAPEC-66", "capec")
    'CAPEC-66'
    """
    if not raw:
        return ""
    s = raw.strip().upper()
    if kind == "mitre":
        # Keep the leading T and any sub-technique suffix.
        m = re.search(r"T\s*\d{4}(?:\.\d{3})?", s)
        if not m:
            return ""
        return m.group(0).replace(" ", "")
    prefix = {"cve": "CVE", "cwe": "CWE", "capec": "CAPEC"}.get(kind)
    if prefix is None:
        return s
    # Pull the numeric (and for CVE, year-number) portion.
    if kind == "cve":
        m = re.search(r"CVE[\s-]*(\d{4})[\s-]*(\d{4,})", s)
        return f"CVE-{m.group(1)}-{m.group(2)}" if m else ""
    m = re.search(rf"{prefix}[\s-]*(\d+)", s)
    return f"{prefix}-{m.group(1)}" if m else ""


def extract_ids(text: str, kind: str) -> set[str]:
    """Extract the set of normalized identifiers of ``kind`` from free text."""
    if not text:
        return set()
    out: set[str] = set()
    for m in ID_PATTERNS[kind].finditer(text):
        norm = normalize_id(m.group(0), kind)
        if norm:
            out.add(norm)
    return out


def prf1(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    """Precision / Recall / F1 over two identifier sets.

    Edge cases follow the standard IR convention:
      - empty gold and empty pred  -> (1, 1, 1)  (correctly predicted nothing)
      - empty pred, non-empty gold -> (0, 0, 0)
      - non-empty pred, empty gold -> (0, 0, 0)
    """
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    inter = len(pred & gold)
    p = inter / len(pred)
    r = inter / len(gold)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


@dataclass
class ItemScore:
    id: str
    task: str
    precision: float
    recall: float
    f1: float
    pred_ids: list[str]
    gold_ids: list[str]
    exact_match: bool


def _gold_id_set(ground_truth: dict) -> tuple[str, set[str], set[str]]:
    """Return (target_type, gold_id_set, valid_id_set) from a ground_truth dict.

    ``gold`` is the canonical answer used for P/R/F1. ``valid`` additionally
    includes any ``valid_target_ids`` — the full set of targets that the
    authoritative source catalog links to this item's source (some items have
    several). ``valid`` always contains ``gold``.
    """
    kind = ground_truth["target_type"]
    if "target_id" in ground_truth and ground_truth.get("target_id"):
        gold = {ground_truth["target_id"].upper()}
    else:
        gold = {g.upper() for g in (ground_truth.get("target_ids") or [])}
    valid = {g.upper() for g in (ground_truth.get("valid_target_ids") or [])} | gold
    return kind, gold, valid


def score_id_item(prediction: str, ground_truth: dict, *, task: str = "",
                  item_id: str = "") -> ItemScore:
    """Score one ID-based item (Entity Linking or Entity Attribution).

    When the item lists multiple valid targets (``valid_target_ids``), any
    non-empty prediction drawn entirely from that valid set counts as correct
    (P=R=F1=1). Otherwise we compute P/R/F1 against the single canonical gold.
    """
    kind, gold, valid = _gold_id_set(ground_truth)
    pred = {p.upper() for p in extract_ids(prediction, kind)}
    if pred and pred <= valid and len(valid) > len(gold):
        return ItemScore(
            id=item_id, task=task,
            precision=1.0, recall=1.0, f1=1.0,
            pred_ids=sorted(pred), gold_ids=sorted(gold),
            exact_match=True,
        )
    p, r, f1 = prf1(pred, gold)
    return ItemScore(
        id=item_id, task=task,
        precision=p, recall=r, f1=f1,
        pred_ids=sorted(pred), gold_ids=sorted(gold),
        exact_match=(pred == gold),
    )


def aggregate_scores(items: list[ItemScore]) -> dict:
    """Mean P/R/F1/exact-match over a list of ItemScores, plus per-task."""
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    by_task: dict[str, list[ItemScore]] = {}
    for it in items:
        by_task.setdefault(it.task, []).append(it)

    per_task = {}
    for task, its in sorted(by_task.items()):
        per_task[task] = {
            "n": len(its),
            "precision": _mean([i.precision for i in its]),
            "recall": _mean([i.recall for i in its]),
            "f1": _mean([i.f1 for i in its]),
            "exact_match": _mean([1.0 if i.exact_match else 0.0 for i in its]),
        }
    overall = {
        "n": len(items),
        "precision": _mean([i.precision for i in items]),
        "recall": _mean([i.recall for i in items]),
        "f1": _mean([i.f1 for i in items]),
        "exact_match": _mean([1.0 if i.exact_match else 0.0 for i in items]),
    }
    return {"overall": overall, "per_task": per_task}


def score_predictions(predictions: dict[str, str], qa_records: list) -> dict:
    """Score ID-based predictions against gold QA records.

    Parameters
    ----------
    predictions : {qa_id: prediction_text}
    qa_records  : list of QA objects (from cticonnect.load_*), each with
                  .id, .task, .eval_type, .ground_truth (GroundTruth or dict)

    Returns a dict with `overall`, `per_task`, and a list of `skipped_judge`
    item ids (Multi-Doc Synthesis items that need the LLM judge instead).
    """
    items: list[ItemScore] = []
    skipped_judge: list[str] = []
    missing: list[str] = []

    for qa in qa_records:
        if qa.eval_type == "judge":
            skipped_judge.append(qa.id)
            continue
        if qa.id not in predictions:
            missing.append(qa.id)
            continue
        gt = qa.ground_truth
        gt_dict = gt if isinstance(gt, dict) else {
            "target_type": gt.target_type,
            "target_id": gt.target_id,
            "target_ids": gt.target_ids,
            "valid_target_ids": gt.valid_target_ids,
        }
        items.append(score_id_item(
            predictions[qa.id], gt_dict, task=qa.task, item_id=qa.id,
        ))

    result = aggregate_scores(items)
    result["skipped_judge"] = skipped_judge
    result["missing_predictions"] = missing
    return result
