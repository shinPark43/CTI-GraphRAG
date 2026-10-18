"""gds.cypher parses into the named steps the driver runs, and projects corpus_kb only."""

from ctiattr.gds import statements


def test_named_steps_in_order():
    names = list(statements())
    assert names == ["drop", "drop_directed", "project", "project_directed", "degree", "wcc", "pagerank",
                     "louvain", "betweenness", "path"]
    assert all(q and not q.endswith(";") for q in statements().values())


def test_report_layer_is_not_projected():
    s = statements()
    for step in ("project", "project_directed"):
        assert not any(w in s[step] for w in ("ReportBehavior", "ReportChunk", "ThreatReport", "GROUNDS"))
