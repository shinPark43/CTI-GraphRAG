"""Citation paths render as Cypher that matches exactly one path."""

from ctiattr.citations import _path, cypher_for
from ctiattr.schemas import AttributionResponse, Citation


def test_cypher_for_report_path():
    path = _path(("ThreatReport", "BLOG-1"), "HAS_CHUNK", ("ReportChunk", "BLOG-1#0"), "HAS_BEHAVIOR",
                 ("ReportBehavior", "BLOG-1#0@7"), "GROUNDS", ("AttackTechnique", "T1486"))
    assert cypher_for(path) == (
        "MATCH p = (:ThreatReport {id: 'BLOG-1'})-[:HAS_CHUNK]->(:ReportChunk {id: 'BLOG-1#0'})"
        "-[:HAS_BEHAVIOR]->(:ReportBehavior {id: 'BLOG-1#0@7'})-[:GROUNDS]->(:AttackTechnique {id: 'T1486'}) RETURN p")


def test_tactics_are_matched_by_name():
    assert "(:Tactic {name: 'impact'})" in cypher_for(_path(("AttackTechnique", "T1486"), "IN_TACTIC",
                                                              ("Tactic", "impact")))


def test_citations_ride_in_the_response_not_the_answer():
    path = _path(("CVE", "CVE-2021-0001"), "HAS_WEAKNESS", ("CWE", "CWE-79"))
    c = Citation(kind="cve", path=path, cypher=cypher_for(path), source="cticonnect:corpus_kb",
                 method="asserted", confidence=1.0, weakness_type="Primary")
    r = AttributionResponse(answer_id="CWE-79", answer_title="XSS", citations=[c], model="m", latency_ms=1.0)
    assert r.citations[0].path[-1].node == "CWE-79" and "CVE-2021-0001" not in r.answer_id
