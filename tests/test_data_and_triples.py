"""Passage extraction, triple scoring, and committed data files against the Pydantic schemas."""

import json

import pytest
from pydantic import ValidationError

from ctiattr.config import ROOT
from ctiattr.data import passage
from ctiattr.schemas import AttributionResponse, GoldTriple, GroundedBehavior, KBNode, ReportChunk
from ctiattr.triples import prf, score_triples

DERIVED = ROOT / "data" / "derived"


@pytest.mark.parametrize("question,expected", [
    ('I recently read in a security blog: "Attackers dumped LSASS memory." Which technique is this?',
     "Attackers dumped LSASS memory."),
    ('I recently read in a security blog: "could lead to DDoS attacks on .NET applications.". Please provide: 1) the CWE',
     "could lead to DDoS attacks on .NET applications."),
    ("No quote in this question at all?", "No quote in this question at all?"),
])
def test_passage_extraction(question, expected):
    assert passage(question) == expected


def test_prf_and_triple_scores():
    gold = {("BLOG-1", "DESCRIBES_TECHNIQUE", "T1059"), ("BLOG-2", "DESCRIBES_WEAKNESS", "CWE-79")}
    pred = {("BLOG-1", "DESCRIBES_TECHNIQUE", "T1059.001"), ("BLOG-2", "DESCRIBES_WEAKNESS", "CWE-79")}
    assert prf(pred, gold)["f1"] == 0.5
    res = score_triples(pred, gold, known_ids={"T1059", "T1059.001", "CWE-79"}, retired=set())
    assert res["diagnostics"]["near_miss_parent_or_child"] == 1
    assert res["per_predicate"]["DESCRIBES_WEAKNESS"]["f1"] == 1.0


def test_schema_rejects_malformed_ids():
    with pytest.raises(ValidationError):
        KBNode(id="T59", label="AttackTechnique", title="x")
    with pytest.raises(ValidationError):
        AttributionResponse(answer_id="CWE79", answer_title="x", model="m", latency_ms=1.0)


def test_report_layer_files_match_schemas():
    chunks = [ReportChunk(**json.loads(l)) for l in (DERIVED / "report_chunks.jsonl").open()]
    behaviors = [GroundedBehavior(**json.loads(l)) for l in (DERIVED / "report_ttps.jsonl").open()]
    by_id = {c.chunk_id: c for c in chunks}
    assert len(chunks) == 650 and len(behaviors) == 3512
    assert all(b.behavior in " ".join(by_id[b.chunk_id].text.split()) for b in behaviors)


def test_gold_triples_match_schema():
    data = json.loads((ROOT / "data" / "gold_triples.json").read_text())
    triples = [GoldTriple(id=t["id"], subject_id=t["subject"]["id"], predicate=t["predicate"],
                          object_id=t["object"]["id"], evidence_text=t["evidence_text"],
                          source_question=t["source_question"]) for t in data["triples"]]
    assert len(triples) >= 50
