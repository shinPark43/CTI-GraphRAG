"""Extraction-confidence signals: name agreement, report spans, tiers, audit precision."""

import csv

from ctiattr.provenance import EDGE_PROVENANCE, audit_precision, name_agreement, span, tier, wilson
from ctiattr.schemas import GraphEdge, Predicate


def test_name_agreement_accepts_parent_child_form():
    assert name_agreement("OS Credential Dumping: LSASS Memory", "LSASS Memory", "OS Credential Dumping") == 1.0
    assert name_agreement("Java", "Python", "Command and Scripting Interpreter") < 0.6


def test_span_finds_phrase_across_whitespace():
    text = "they executed  obfuscated\nPowerShell to persist"
    assert span(text, 100, "executed obfuscated PowerShell") == (105, 136)
    assert span(text, 0, "not there") is None


def test_tiers():
    assert tier(1.0, 3) == "supported"
    assert tier(0.9, 51) == "context"
    assert tier(0.2, 1) == "conflict"


def test_audit_precision_by_tier(tmp_path):
    p = tmp_path / "audit.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, ["tier", "correct"])
        w.writeheader()
        w.writerows([{"tier": "supported", "correct": c} for c in "1110"] + [{"tier": "context", "correct": ""}])
    out = audit_precision(p)
    assert out["supported"]["precision"] == 0.75 and "context" not in out
    lo, hi = wilson(3, 4)
    assert lo < 0.75 < hi


def test_every_relationship_type_has_provenance():
    loaded = {p.value for p in Predicate} - {"DESCRIBES_TECHNIQUE", "DESCRIBES_WEAKNESS"}
    assert loaded <= set(EDGE_PROVENANCE)
    e = GraphEdge(source_id="T1003.001", target_id="T1003", predicate=Predicate.SUBTECHNIQUE_OF,
                  origin="corpus_kb", method="derived", confidence=1.0)
    assert e.tier is None
