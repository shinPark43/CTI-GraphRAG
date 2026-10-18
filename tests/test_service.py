"""Service helpers that need no model or database: taxonomy guess, ID parsing, path checks."""

import pytest

from ctiattr.schemas import AttributionResponse, PathStep, Taxonomy
from ctiattr.service import AttributionService, first_id, guess_taxonomy


def test_guess_taxonomy():
    assert guess_taxonomy("Which CWE maps to this vulnerability?") == Taxonomy.CWE
    assert guess_taxonomy("Attackers dumped LSASS memory with Mimikatz") == Taxonomy.ATTACK


def test_first_id_keeps_text_order():
    assert first_id("Not T1003 but T1003.001; see T1059", "mitre") == "T1003"
    assert first_id("maps to cwe-79", "cwe") == "CWE-79"
    assert first_id("no identifier here", "cwe") is None


def test_response_allows_a_baseline_without_an_id():
    assert AttributionResponse(answer_id=None, model="m", latency_ms=1.0).answer_id is None


class FakeGraph:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return [{"n": 1}]


def test_verify_parameterizes_ids_and_rejects_unknown_schema():
    svc = AttributionService.__new__(AttributionService)  # no model or embedder needed
    svc.graph = FakeGraph()
    path = [PathStep(label="CVE", node="CVE-1' DETACH DELETE n //"), PathStep(rel="HAS_WEAKNESS"),
            PathStep(label="CWE", node="CWE-79")]
    assert svc.verify(path) == 1
    query, params = svc.graph.calls[0]
    assert "DETACH" not in query and params["n0"].startswith("CVE-1'")
    with pytest.raises(ValueError):
        svc.verify([PathStep(label="CVE", node="x"), PathStep(rel="DELETE"), PathStep(label="CWE", node="y")])
