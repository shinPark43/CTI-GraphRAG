"""Load CTIConnect attribution questions and knowledge-base entries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ctiattr.config import RAW

TASKS = ("ata", "vca")

# Which knowledge base each task retrieves against (== ground_truth.target_type).
TASK_KB = {"ata": "mitre", "vca": "cwe"}

_KB_ID_FIELD = {"cwe": "cwe_id", "mitre": "mitre_id", "capec": "capec_id", "cve": "cve_id"}
_KB_PREFIX = {"cwe": "CWE-", "capec": "CAPEC-", "cve": "CVE-", "mitre": "T"}


@dataclass(frozen=True)
class QA:
    id: str
    task: str
    question: str
    ground_truth: dict
    report_id: str  # source report (e.g. "BLOG-10"); the unit for splitting

    @property
    def gold(self) -> list[str]:
        return [g.upper() for g in self.ground_truth["target_ids"]]


@dataclass(frozen=True)
class KBEntry:
    doc_id: str  # canonical id, e.g. "CWE-79", "T1059.001"
    title: str
    text: str    # the raw `contents` JSON string, as CTIConnect embeds it


def load_task(task: str, raw: Path = RAW) -> list[QA]:
    out = []
    with open(raw / "data" / "entity_attribution" / f"{task}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(QA(d["id"], d["task"], d["question"], d["ground_truth"],
                          d["source"]["source_id"]))
    return out


def load_all(raw: Path = RAW) -> list[QA]:
    return [qa for t in TASKS for qa in load_task(t, raw)]


def canonical_id(kind: str, raw_id: str) -> str:
    raw_id = str(raw_id)
    prefix = _KB_PREFIX[kind]
    return raw_id if raw_id.upper().startswith(prefix) else prefix + raw_id


def load_kb(kind: str, raw: Path = RAW) -> list[KBEntry]:
    out = []
    with open(raw / "corpus_kb" / f"{kind}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            raw_id = d.get(_KB_ID_FIELD[kind]) or d.get("id")
            out.append(KBEntry(canonical_id(kind, raw_id), d.get("title") or "",
                               str(d.get("contents") or "")))
    return out
