"""Neo4j storage for the snapshot graph and report layer, and the SGS bridges as Cypher.

The loader writes exactly the structure src/ctiattr/snapshot.py derives in
Python (same IDs, same edges), plus the report layer and the cached
text-embedding-3-large vectors. The query methods return the same rankings
the Python candidate generators compute, so either backend can feed SGS.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable

import numpy as np
from neo4j import GraphDatabase

from ctiattr.config import ROOT, RAW
from ctiattr.retrieval import Embedder
from ctiattr.snapshot import Snapshot

SCHEMA = ROOT / "schema.cypher"
DERIVED = ROOT / "data" / "derived"
BATCH = 200


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _batches(rows: list[dict], n: int = BATCH) -> Iterable[list[dict]]:
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


class GraphDB:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"), password or os.environ["NEO4J_PASSWORD"]))

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **params):
        records, _, _ = self.driver.execute_query(query, params)
        return records

    # --- schema --------------------------------------------------------------

    def apply_schema(self) -> int:
        text = "\n".join(l for l in SCHEMA.read_text().splitlines() if not l.lstrip().startswith("//"))
        statements = [s.strip() for s in text.split(";") if s.strip()]
        for s in statements:
            self.run(s)
        return len(statements)

    def wipe(self) -> None:
        # CALL { } IN TRANSACTIONS needs an implicit (auto-commit) transaction
        with self.driver.session() as session:
            session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()

    def _write(self, query: str, rows: list[dict]) -> None:
        for chunk in _batches(rows):
            self.run(query, rows=chunk)

    # --- load ----------------------------------------------------------------

    def load(self, s: Snapshot, emb: Embedder) -> dict:
        tech_vec = emb.embed([f"{e.title}. {e.text}".strip() for e in s.mitre])
        self._write("""UNWIND $rows AS r MERGE (t:AttackTechnique {id: r.id})
            SET t.title = r.title, t.title_norm = r.norm, t.text = r.text, t.domain = r.domain,
                t.tactics = r.tactics, t.is_subtechnique = r.sub, t.embedding = r.vec""",
                    [{"id": e.doc_id, "title": e.title, "norm": _norm(e.title), "text": e.text,
                      "domain": s.domain(e.doc_id), "tactics": s.tactics(e.doc_id), "sub": "." in e.doc_id,
                      "vec": tech_vec[i].tolist()} for i, e in enumerate(s.mitre)])
        self._write("""UNWIND $rows AS r MATCH (c:AttackTechnique {id: r.child}), (p:AttackTechnique {id: r.parent})
            MERGE (c)-[:SUBTECHNIQUE_OF]->(p)""",
                    [{"child": t, "parent": s.parent(t)} for t in s.technique_ids
                     if s.parent(t) in s.technique_ids])
        self._write("""UNWIND $rows AS r MERGE (x:Tactic {name: r.tactic}) WITH r, x
            MATCH (t:AttackTechnique {id: r.id}) MERGE (t)-[:IN_TACTIC]->(x)""",
                    [{"id": t, "tactic": a} for t in s.technique_ids for a in s.tactics(t)])
        self._write("""UNWIND $rows AS r MATCH (o:AttackTechnique {id: r.old}), (n:AttackTechnique {id: r.new})
            MERGE (o)-[:REPLACED_BY]->(n) SET o.replaced_by = r.new""",
                    [{"old": o, "new": n} for o, n in s.retired.items()])

        cwe_vec = emb.embed([f"{e.title}. {e.text}".strip() for e in s.cwe])
        self._write("""UNWIND $rows AS r MERGE (w:CWE {id: r.id})
            SET w.title = r.title, w.title_norm = r.norm, w.text = r.text, w.embedding = r.vec""",
                    [{"id": e.doc_id.upper(), "title": e.title, "norm": _norm(e.title), "text": e.text,
                      "vec": cwe_vec[i].tolist()} for i, e in enumerate(s.cwe)])

        capec_vec = emb.embed([f"{c.name}. {c.description}" for c in s.capec])
        self._write("""UNWIND $rows AS r MERGE (c:CAPEC {id: r.id})
            SET c.name = r.name, c.description = r.description, c.embedding = r.vec""",
                    [{"id": c.id, "name": c.name, "description": c.description, "vec": capec_vec[i].tolist()}
                     for i, c in enumerate(s.capec)])
        self._write("""UNWIND $rows AS r MATCH (c:CAPEC {id: r.capec}), (w:CWE {id: r.cwe})
            MERGE (c)-[:EXPLOITS_WEAKNESS]->(w)""",
                    [{"capec": c.id, "cwe": w} for c in s.capec for w in c.cwes])
        self._write("""UNWIND $rows AS r MATCH (c:CAPEC {id: r.capec}), (t:AttackTechnique {id: r.tech})
            MERGE (c)-[:MAPS_TO_TECHNIQUE]->(t)""",
                    [{"capec": c.id, "tech": t} for c in s.capec for t in c.techniques])

        cve_vec = emb.embed([c.description for c in s.cve])
        self._write("""UNWIND $rows AS r MERGE (v:CVE {id: r.id})
            SET v.description = r.description, v.embedding = r.vec""",
                    [{"id": c.id, "description": c.description, "vec": cve_vec[i].tolist()}
                     for i, c in enumerate(s.cve)])
        self._write("""UNWIND $rows AS r MATCH (v:CVE {id: r.cve}), (w:CWE {id: r.cwe})
            MERGE (v)-[:HAS_WEAKNESS]->(w)""",
                    [{"cve": c.id, "cwe": w} for c in s.cve for w in c.cwes])

        reports = {}
        with open(RAW / "corpus_reports" / "preprocessed_reports.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                reports[f"BLOG-{r['id']}"] = r
        self._write("""UNWIND $rows AS r MERGE (x:ThreatReport {id: r.id})
            SET x.title = r.title, x.url = r.url, x.published = r.published""",
                    [{"id": k, "title": r.get("title", ""), "url": r.get("link", ""),
                      "published": r.get("publish_date", "")} for k, r in reports.items()])
        chunks = [json.loads(l) for l in (DERIVED / "report_chunks.jsonl").open()]
        self._write("""UNWIND $rows AS r MATCH (x:ThreatReport {id: r.report_id})
            MERGE (k:ReportChunk {id: r.chunk_id})
            SET k.report_id = r.report_id, k.start = r.start, k.end = r.end, k.text = r.text
            MERGE (x)-[:HAS_CHUNK]->(k)""", chunks)
        ttps = [json.loads(l) for l in (DERIVED / "report_ttps.jsonl").open()]
        beh_vec = emb.embed([b["behavior"] for b in ttps])
        self._write("""UNWIND $rows AS r MATCH (k:ReportChunk {id: r.chunk_id}), (t:AttackTechnique {id: r.tech})
            MERGE (b:ReportBehavior {id: r.id})
            SET b.behavior = r.behavior, b.report_id = r.report_id, b.embedding = r.vec
            MERGE (k)-[:HAS_BEHAVIOR]->(b) MERGE (b)-[:GROUNDS]->(t)""",
                    [{"id": f"{b['chunk_id']}@{i}", "chunk_id": b["chunk_id"], "report_id": b["report_id"],
                      "behavior": b["behavior"], "tech": b["technique_id"], "vec": beh_vec[i].tolist()}
                     for i, b in enumerate(ttps)])
        self.run("CALL db.awaitIndexes(600)")
        return self.counts()

    def counts(self) -> dict:
        nodes = {r["label"]: r["n"] for r in self.run(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")}
        rels = {r["type"]: r["n"] for r in self.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n")}
        return {"nodes": nodes, "relationships": rels}

    # --- SGS bridges as Cypher ----------------------------------------------
    #
    # exact=True (default, used for evaluation) scores every node with
    # vector.similarity.cosine, so rankings match the in-memory implementation.
    # exact=False uses the HNSW vector indexes: faster, but approximate, so
    # near-ties can reorder (see scripts/check_graph_bridges.py).
    # Neo4j reports cosine similarity as (1 + cos) / 2; sums convert back with 2s - 1.

    _LABEL = {"technique_emb": "AttackTechnique", "cwe_emb": "CWE", "capec_emb": "CAPEC",
              "cve_emb": "CVE", "behavior_emb": "ReportBehavior"}

    def _nearest(self, index: str, var: str, k: int, exact: bool, where: str = "") -> str:
        """Cypher fragment binding `var` and `score` for the k nodes nearest $v."""
        if exact:
            cond = f"WHERE {where}" if where else ""
            return (f"MATCH ({var}:{self._LABEL[index]}) {cond} "
                    f"WITH {var}, vector.similarity.cosine({var}.embedding, $v) AS score "
                    f"ORDER BY score DESC LIMIT {int(k)}")
        cond = f"WHERE {where}" if where else ""
        fetch = int(k) * 4 if where else int(k)
        return (f"CALL db.index.vector.queryNodes('{index}', {fetch}, $v) YIELD node AS {var}, score {cond} "
                f"WITH {var}, score ORDER BY score DESC LIMIT {int(k)}")

    def dense(self, index: str, vector: np.ndarray, k: int, exact: bool = True) -> list[str]:
        rows = self.run(self._nearest(index, "n", k, exact) + " RETURN n.id AS id", v=vector.tolist())
        return [r["id"] for r in rows]

    def capec_bridge(self, vector: np.ndarray, n: int = 10, exact: bool = True) -> list[str]:
        """Rule R4: passage ~ CAPEC description -> EXPLOITS_WEAKNESS -> CWE."""
        rows = self.run(self._nearest("capec_emb", "c", n, exact) + """
            MATCH (c)-[:EXPLOITS_WEAKNESS]->(w:CWE)
            RETURN w.id AS id, sum(2 * score - 1) AS s ORDER BY s DESC""", v=vector.tolist())
        return [r["id"] for r in rows]

    def cve_vote(self, vector: np.ndarray, n: int = 20, exact: bool = True) -> list[str]:
        """Rule R5 (VCA): passage ~ CVE description -> HAS_WEAKNESS -> CWE."""
        rows = self.run(self._nearest("cve_emb", "x", n, exact) + """
            MATCH (x)-[:HAS_WEAKNESS]->(w:CWE)
            RETURN w.id AS id, sum(2 * score - 1) AS s ORDER BY s DESC""", v=vector.tolist())
        return [r["id"] for r in rows]

    def report_vote(self, vector: np.ndarray, report_id: str, n: int = 30, exact: bool = True) -> list[str]:
        """Rule R6 (ATA): nearest behaviors from other reports -> GROUNDS -> technique."""
        rows = self.run(self._nearest("behavior_emb", "b", n, exact, where="b.report_id <> $rid") + """
            MATCH (b)-[:GROUNDS]->(t:AttackTechnique)
            RETURN t.id AS id, sum(2 * score - 1) AS s ORDER BY s DESC""",
                        rid=report_id, v=vector.tolist())
        return [r["id"] for r in rows]
