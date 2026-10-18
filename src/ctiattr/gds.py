"""Graph Data Science over the corpus_kb graph in Neo4j (gds.cypher).

run_all() projects the graph and runs degree, WCC, PageRank, Louvain and
betweenness, writing each back as a node property. The summary queries below
read those properties back for the topology report (scripts/gds_analysis.py).
scores() hands PageRank and community to the retrieval experiments.
"""

from __future__ import annotations

import re

from ctiattr.config import ROOT
from ctiattr.graph_db import GraphDB

GDS_CYPHER = ROOT / "gds.cypher"
KB_LABELS = ("AttackTechnique", "Tactic", "CWE", "CAPEC", "CVE")
_NODE = "n:AttackTechnique OR n:Tactic OR n:CWE OR n:CAPEC OR n:CVE"


def statements() -> dict[str, str]:
    """The `// name:` blocks of gds.cypher, comments stripped."""
    out = {}
    for block in re.split(r"^// name: ", GDS_CYPHER.read_text(), flags=re.M)[1:]:
        name, body = block.split("\n", 1)
        body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("//"))
        out[name.strip()] = body.strip().rstrip(";")
    return out


def run_all(db: GraphDB) -> dict[str, dict]:
    """Project and run every algorithm; returns each step's YIELD row."""
    out = {}
    for name, q in statements().items():
        if name == "path":
            continue
        rows = db.run(q)
        out[name] = rows[0].data() if rows else {}
    return out


def shortest_path(db: GraphDB, source: str, target: str) -> dict | None:
    rows = db.run(statements()["path"], source=source, target=target)
    return rows[0].data() if rows else None


def node_id(record_labels: list[str], nid: str | None, name: str | None) -> str:
    return f"tactic:{name}" if "Tactic" in record_labels else nid


def scores(db: GraphDB) -> dict[str, dict]:
    """PageRank, community, degree and betweenness per node, keyed like graph_embed node ids."""
    rows = db.run(f"MATCH (n) WHERE {_NODE} RETURN labels(n) AS l, n.id AS id, n.name AS name, "
                  "n.pagerank AS pr, n.community AS c, n.degree AS d, n.betweenness AS b")
    return {node_id(r["l"], r["id"], r["name"]): {"pagerank": r["pr"], "community": r["c"], "degree": r["d"],
                                                 "betweenness": r["b"]} for r in rows}


# --- summaries --------------------------------------------------------------

def degree_by_label(db: GraphDB) -> dict[str, dict]:
    rows = db.run(f"""MATCH (n) WHERE {_NODE} WITH labels(n)[0] AS label, n.degree AS d
        RETURN label, count(*) AS nodes, avg(d) AS mean, percentileDisc(d, 0.5) AS median, max(d) AS max,
               sum(CASE WHEN d = 0 THEN 1 ELSE 0 END) AS isolated""")
    return {r["label"]: {k: r[k] for k in ("nodes", "mean", "median", "max", "isolated")} for r in rows}


def components(db: GraphDB, top: int = 5) -> list[dict]:
    rows = db.run(f"""MATCH (n) WHERE {_NODE} WITH n.component AS c, labels(n)[0] AS label, count(*) AS k
        WITH c, collect([label, k]) AS mix, sum(k) AS size ORDER BY size DESC LIMIT $top
        RETURN c, size, mix""", top=top)
    return [{"component": r["c"], "size": r["size"], "labels": dict(r["mix"])} for r in rows]


def top_nodes(db: GraphDB, prop: str, label: str, k: int = 10) -> list[dict]:
    rows = db.run(f"""MATCH (n:{label}) RETURN coalesce(n.id, n.name) AS id, coalesce(n.title, n.name) AS title,
        n.{prop} AS score ORDER BY score DESC LIMIT $k""", k=k)
    return [{"id": r["id"], "title": r["title"], prop: round(r["score"], 4)} for r in rows]


def communities(db: GraphDB, top: int = 10) -> list[dict]:
    """Largest Louvain communities: size, label mix, and their highest-PageRank members."""
    rows = db.run(f"""MATCH (n) WHERE {_NODE}
        WITH n.community AS c, labels(n)[0] AS label, n ORDER BY n.pagerank DESC
        WITH c, label, count(*) AS k, collect(coalesce(n.id, n.name) + ' ' + coalesce(n.title, n.name, ''))[..3] AS lead
        WITH c, collect([label, k]) AS mix, sum(k) AS size, collect([label, lead]) AS leads
        ORDER BY size DESC LIMIT $top RETURN c, size, mix, leads""", top=top)
    return [{"community": r["c"], "size": r["size"], "labels": dict(r["mix"]), "leaders": dict(r["leads"])}
            for r in rows]


def community_purity(db: GraphDB) -> dict:
    """How well Louvain communities line up with ATT&CK tactics: share of techniques whose
    community's most common tactic is one of their own tactics."""
    row = db.run("""MATCH (t:AttackTechnique)-[:IN_TACTIC]->(x:Tactic)
        WITH t.community AS c, x.name AS tactic, count(*) AS k ORDER BY k DESC
        WITH c, collect(tactic)[0] AS top
        MATCH (t:AttackTechnique {community: c})-[:IN_TACTIC]->(x:Tactic)
        WITH t, collect(x.name) AS own, top
        RETURN count(t) AS techniques, sum(CASE WHEN top IN own THEN 1 ELSE 0 END) AS aligned""")[0]
    return {"techniques": row["techniques"], "aligned": row["aligned"],
            "share": round(row["aligned"] / max(row["techniques"], 1), 3)}


def bridges(db: GraphDB) -> dict:
    """Coverage of the CVE -> CWE <- CAPEC -> technique bridge that links the two tasks."""
    q = lambda s: db.run(s)[0]["n"]
    return {
        "cwe_total": q("MATCH (w:CWE) RETURN count(w) AS n"),
        "cwe_with_cve": q("MATCH (w:CWE) WHERE EXISTS { (:CVE)-[:HAS_WEAKNESS]->(w) } RETURN count(w) AS n"),
        "cwe_with_capec": q("MATCH (w:CWE) WHERE EXISTS { (:CAPEC)-[:EXPLOITS_WEAKNESS]->(w) } RETURN count(w) AS n"),
        "cwe_reaching_technique": q("""MATCH (w:CWE) WHERE EXISTS {
            (w)<-[:EXPLOITS_WEAKNESS]-(:CAPEC)-[:MAPS_TO_TECHNIQUE]->(:AttackTechnique) } RETURN count(w) AS n"""),
        "technique_total": q("MATCH (t:AttackTechnique) RETURN count(t) AS n"),
        "technique_with_capec": q("""MATCH (t:AttackTechnique) WHERE EXISTS { (:CAPEC)-[:MAPS_TO_TECHNIQUE]->(t) }
            RETURN count(t) AS n"""),
        "cve_to_technique_paths": q("""MATCH (:CVE)-[:HAS_WEAKNESS]->(:CWE)<-[:EXPLOITS_WEAKNESS]-(:CAPEC)
            -[:MAPS_TO_TECHNIQUE]->(:AttackTechnique) RETURN count(*) AS n"""),
    }


def example_bridge(db: GraphDB) -> tuple[str, str] | None:
    """A CVE and a technique joined through CWE and CAPEC, for the path demo."""
    rows = db.run("""MATCH (v:CVE)-[:HAS_WEAKNESS]->(w:CWE)<-[:EXPLOITS_WEAKNESS]-(:CAPEC)-[:MAPS_TO_TECHNIQUE]->(t)
        RETURN v.id AS v, t.id AS t ORDER BY w.pagerank DESC, v.id LIMIT 1""")
    return (rows[0]["v"], rows[0]["t"]) if rows else None
