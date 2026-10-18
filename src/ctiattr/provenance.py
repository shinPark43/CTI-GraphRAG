"""Provenance and confidence for every relationship in the graph.

Each edge records where it came from (`source`, a :Source node id), how it was
made (`method`), and a `confidence`:

    asserted   stated in a corpus_kb record (kill_chain_phases, CAPEC Related_Weaknesses
               and Taxonomy_Mappings, NVD weaknesses)                      confidence 1.0
    derived    a deterministic rule over corpus_kb (ID pattern, retired-ID title match)
                                                                            confidence 1.0
    segmented  chunking of a report (HAS_CHUNK, HAS_BEHAVIOR)              confidence 1.0
    extracted  the LLM report layer (GROUNDS)                              confidence from the audit

Every GROUNDS edge already passed two hard checks (ID exists in the snapshot,
behavior phrase occurs verbatim in its chunk). Two soft signals then sort it into a tier:

    name_agreement  the technique name the model wrote vs the official title of the ID
                    it gave (difflib ratio, "Parent: Child" accepted); low = likely wrong ID
    sem_rank        rank of the technique among all 1,076 for the behavior phrase alone
                    (text-embedding-3-large cosine); high = the phrase does not say it

    supported  name_agreement >= 0.6 and sem_rank <= 50
    context    name_agreement >= 0.6 and sem_rank > 50   (right ID, meaning comes from context)
    conflict   name_agreement < 0.6                       (ID and name disagree)

A tier's confidence is its precision in the hand audit (data/derived/extraction_audit.csv,
scripts/extraction_audit.py); without an audit it stays unset.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
from dataclasses import dataclass

import numpy as np

from ctiattr.config import RAW, ROOT

AUDIT = ROOT / "data" / "derived" / "extraction_audit.csv"
NAME_OK = 0.6
RANK_OK = 50
TIERS = ("supported", "context", "conflict")

# (source id, method, how) for each relationship type
EDGE_PROVENANCE = {
    "SUBTECHNIQUE_OF":   ("cticonnect:corpus_kb", "derived", "ID pattern T####.### -> T####"),
    "REPLACED_BY":       ("cticonnect:corpus_kb", "derived", "retired Enterprise ID whose title equals a sub-technique's"),
    "IN_TACTIC":         ("cticonnect:corpus_kb", "asserted", "mitre.jsonl kill_chain_phases"),
    "EXPLOITS_WEAKNESS": ("cticonnect:corpus_kb", "asserted", "capec.jsonl Related_Weaknesses"),
    "MAPS_TO_TECHNIQUE": ("cticonnect:corpus_kb", "asserted", "capec.jsonl Taxonomy_Mappings (ATTACK)"),
    "HAS_WEAKNESS":      ("cticonnect:corpus_kb", "asserted", "cve.jsonl NVD weaknesses"),
    "HAS_CHUNK":         ("report_layer", "segmented", "clause-boundary chunking, 200-900 chars"),
    "HAS_BEHAVIOR":      ("report_layer", "segmented", "verbatim behavior span inside the chunk"),
    "GROUNDS":           ("report_layer", "extracted", "LLM: CTIFoundry TTP prompt, validated ID + verbatim"),
}


def sources() -> list[dict]:
    """The :Source nodes: the pinned CTIConnect snapshot and the report-layer build."""
    manifest = json.loads((RAW / "corpus_kb" / "MANIFEST.json").read_text())
    build = json.loads((ROOT / "data" / "derived" / "report_layer_build.json").read_text())
    commit = (RAW / "COMMIT").read_text().strip()
    return [
        {"id": "cticonnect:corpus_kb", "name": "CTIConnect corpus_kb (MITRE ATT&CK, CWE, CAPEC; NVD CVE)",
         "url": "https://github.com/peng-gao-lab/CTIConnect", "commit": commit,
         "snapshot_date": manifest.get("snapshot_date", "")},
        {"id": "cticonnect:corpus_reports", "name": "CTIConnect corpus_reports (preprocessed threat-report summaries)",
         "url": "https://github.com/peng-gao-lab/CTIConnect", "commit": commit, "snapshot_date": ""},
        {"id": "report_layer", "name": "LLM-extracted ATT&CK behaviors grounded in report chunks",
         "url": "", "commit": "", "snapshot_date": build.get("built_on", ""), "model": build["model"],
         "prompt": "CTIFoundry App. G.3 TTP prompt", "chunks": build["chunks"], "extracted": build["extracted"],
         "kept": build["kept"]},
    ]


_norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()


def name_agreement(name: str, title: str, parent_title: str | None = None) -> float:
    """How well a free-text technique name matches the official title (1.0 = same)."""
    opts = [title] + ([f"{parent_title}: {title}"] if parent_title else [])
    return round(max(difflib.SequenceMatcher(None, _norm(name), _norm(o)).ratio() for o in opts), 3)


def span(chunk_text: str, chunk_start: int, phrase: str) -> tuple[int, int] | None:
    """Character offsets of a (whitespace-normalized) phrase in the report text."""
    pat = r"\s+".join(map(re.escape, phrase.split()))
    m = re.search(pat, chunk_text)
    return (chunk_start + m.start(), chunk_start + m.end()) if m else None


def tier(agreement: float, rank: int) -> str:
    if agreement < NAME_OK:
        return "conflict"
    return "supported" if rank <= RANK_OK else "context"


@dataclass
class ExtractionSignals:
    name_agreement: float
    sem_rank: int
    sem_cos: float
    tier: str
    start: int | None
    end: int | None


def extraction_signals(snapshot, emb, ttps: list[dict], chunks: dict[str, dict]) -> list[ExtractionSignals]:
    """Soft signals and report offsets for each report_ttps.jsonl record (embeddings come from the cache)."""
    tids = [e.doc_id for e in snapshot.mitre]
    title = {e.doc_id: e.title for e in snapshot.mitre}
    pos = {t: i for i, t in enumerate(tids)}
    T = emb.embed([f"{e.title}. {e.text}".strip() for e in snapshot.mitre])
    B = emb.embed([b["behavior"] for b in ttps])
    out = []
    for b, v in zip(ttps, B):
        t = b["technique_id"]
        sims = T @ v
        rank = int((sims > sims[pos[t]]).sum()) + 1
        agree = name_agreement(b["technique_name"], title[t], title.get(snapshot.parent(t) or ""))
        c = chunks[b["chunk_id"]]
        s = span(c["text"], c["start"], b["behavior"])
        out.append(ExtractionSignals(agree, rank, round(float(sims[pos[t]]), 4), tier(agree, rank),
                                     *(s or (None, None))))
    return out


def audit_precision(path=AUDIT) -> dict[str, dict]:
    """Per-tier precision from the labelled audit: correct / labelled, with a Wilson 95% interval."""
    if not path.exists():
        return {}
    rows = [r for r in csv.DictReader(path.open()) if r.get("correct", "").strip() in ("0", "1")]
    out = {}
    for t in TIERS:
        ys = [int(r["correct"]) for r in rows if r["tier"] == t]
        if ys:
            out[t] = {"n": len(ys), "correct": sum(ys), "precision": round(sum(ys) / len(ys), 3),
                      "ci95": [round(x, 3) for x in wilson(sum(ys), len(ys))]}
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
