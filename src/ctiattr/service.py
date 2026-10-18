"""Attribution service: one request in, one AttributionResponse out (design document, Section 8).

Used in-process by the Streamlit chatbot (app/streamlit_app.py) and over HTTP by
the FastAPI app (ctiattr.api). SGS answers carry graph citations and a latency
breakdown; vanilla RAG and closed-book are offered for side-by-side comparison
and carry neither. Graph analytics (PageRank, Louvain community) come from the
node properties scripts/gds_analysis.py wrote into Neo4j; without Neo4j the
service still answers, with in-memory retrieval.
"""

from __future__ import annotations

import re
import time

from ctiattr.baselines import closed_book, vanilla_rag
from ctiattr.config import EmbedConfig, LLMConfig
from ctiattr.data import QA, TASK_KB, load_all
from ctiattr.llm import ChatLLM
from ctiattr.retrieval import Embedder, Retriever
from ctiattr.provenance import EDGE_PROVENANCE
from ctiattr.schemas import (AttributionRequest, AttributionResponse, Candidate, Citation, LLMCall,
                             PathStep, Taxonomy)
from ctiattr.sgs import KIND, SGS
from ctiattr.split import load_split
from ctiattr.vendor.cticonnect_metrics import ID_PATTERNS, normalize_id

TEMPLATE = {
    "ata": ('I recently read in a security blog: "{passage}" Please identify the MITRE ATT&CK category for '
            "this behavior, explain the mapping to that category, and also specify which operating systems "
            "are affected (e.g., Linux, macOS, Windows, ESXi) based on the provided MITRE reference."),
    "vca": ('I recently read in a security blog: "{passage}" Please provide: 1) the CWE category that maps '
            "to this vulnerability, 2) an explanation of how this vulnerability maps to the CWE, and 3) what "
            "are the common consequences in terms of business risks (e.g., data exposure, account takeover)."),
}
LABELS = {"AttackTechnique", "Tactic", "CWE", "CAPEC", "CVE", "ThreatReport", "ReportChunk", "ReportBehavior"}
_VCA_HINT = re.compile(r"\b(CWE|weakness|vulnerabilit|CVE-\d)", re.I)


def guess_taxonomy(text: str) -> Taxonomy:
    """CWE if the text talks about weaknesses or vulnerabilities, ATT&CK otherwise."""
    return Taxonomy.CWE if _VCA_HINT.search(text) else Taxonomy.ATTACK


def first_id(text: str, kind: str) -> str | None:
    m = ID_PATTERNS[kind].search(text or "")
    return normalize_id(m.group(0), kind) if m else None


class AttributionService:
    def __init__(self, model: str | None = None, graph=None):
        """graph: an optional ctiattr.graph_db.GraphDB for analytics and path checks."""
        self.emb = Embedder(EmbedConfig.from_env())
        self.llm = ChatLLM(LLMConfig.from_env(model))
        self.sgs = SGS(self.emb)
        self.graph = graph
        self._retriever: Retriever | None = None

    @property
    def model(self) -> str:
        return self.llm.model

    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.emb)
            for kind in set(TASK_KB.values()):
                self._retriever.index(kind)
        return self._retriever

    # --- answering ---------------------------------------------------------

    def attribute(self, req: AttributionRequest) -> AttributionResponse:
        t0 = time.perf_counter()
        task = "ata" if req.taxonomy == Taxonomy.ATTACK else "vca"
        question = req.question or TEMPLATE[task].format(passage=req.passage.strip())
        qa = QA(id="chat", task=task, question=question, ground_truth={"target_ids": []},
                report_id=req.report_id or "")
        if req.method == "sgs":
            text, trace = self.sgs.answer(self.llm, qa, explain=req.explain)
            chosen = trace["chosen"].upper()
            resp = AttributionResponse(
                answer_id=chosen, answer_title=self.sgs.title.get(chosen, ""),
                explanation=trace.get("explanation", ""),
                candidates=[Candidate(id=c.upper(), rank=i + 1, card=trace["cards"][c],
                                      sources=["dense_question", "dense_passage",
                                               "report_vote" if task == "ata" else "capec+cve"])
                            for i, c in enumerate(trace["candidates"])],
                votes=trace["picks"] if task == "vca" else [],
                calls=[LLMCall(step="select", **c) for c in trace["calls"][:len(trace["picks"])]]
                + [LLMCall(step="explain", **c) for c in trace["calls"][len(trace["picks"]):]],
                citations=[Citation(**c) for c in trace.get("citations", [])],
                latency=trace["latency_s"], model=self.model,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            return resp
        if req.method == "vr":
            text, trace = vanilla_rag(qa, {"llm": self.llm, "k": 5, "retriever": self.retriever()})
        else:
            text, trace = closed_book(qa, {"llm": self.llm})
        aid = first_id(text, KIND[task])
        total = time.perf_counter() - t0
        return AttributionResponse(
            answer_id=aid, answer_title=self.sgs.title.get(aid or "", ""), explanation=text,
            calls=[LLMCall(step="answer", **c) for c in trace["calls"]],
            latency={"llm": round(sum(c["latency_s"] for c in trace["calls"]), 3), "total": round(total, 3)},
            model=self.model, latency_ms=round(total * 1000, 1))

    # --- graph -------------------------------------------------------------

    def graph_context(self, node_id: str) -> dict | None:
        """GDS outputs for a node: PageRank and its percentile within the label, Louvain community."""
        if self.graph is None or not node_id:
            return None
        rows = self.graph.run("""
            MATCH (n {id: $id}) WHERE n:AttackTechnique OR n:CWE
            WITH n, labels(n)[0] AS label
            CALL (n, label) {
              MATCH (m) WHERE label IN labels(m) AND m.pagerank IS NOT NULL
              RETURN avg(CASE WHEN m.pagerank < n.pagerank THEN 1.0 WHEN m.pagerank = n.pagerank THEN 0.5 ELSE 0.0 END) AS pct }
            CALL (n) {
              MATCH (m {community: n.community}) WHERE m.pagerank IS NOT NULL AND m <> n
              WITH m ORDER BY m.pagerank DESC LIMIT 5
              RETURN collect(coalesce(m.id, m.name) + ' ' + coalesce(m.title, m.name, '')) AS peers,
                     count(*) AS shown }
            CALL (n) { MATCH (m {community: n.community}) RETURN count(m) AS size }
            RETURN label, n.pagerank AS pagerank, pct, n.degree AS degree, n.betweenness AS betweenness,
                   n.community AS community, size, peers""", id=node_id)
        return rows[0].data() if rows else None

    def verify(self, path: list[PathStep]) -> int | None:
        """How many copies of a citation path exist in the live graph (1 = verified).

        Labels and relationship types are checked against the schema and node ids are
        passed as parameters, so the query can only ever match."""
        if self.graph is None:
            return None
        parts, params = [], {}
        for i, step in enumerate(path):
            if step.node is not None:
                if step.label not in LABELS:
                    raise ValueError(f"unknown label {step.label!r}")
                key = "name" if step.label == "Tactic" else "id"
                parts.append(f"(:{step.label} {{{key}: $n{i}}})")
                params[f"n{i}"] = step.node
            elif step.rel in EDGE_PROVENANCE:
                parts.append(f"-[:{step.rel}]->")
            else:
                raise ValueError(f"unknown relationship {step.rel!r}")
        return self.graph.run(f"MATCH p = {''.join(parts)} RETURN count(p) AS n", **params)[0]["n"]

    def stats(self) -> dict | None:
        if self.graph is None:
            return None
        return self.graph.counts()


def dev_examples() -> list[QA]:
    """Example questions for the demo: development split only (the test split stays untouched)."""
    dev = set(load_split()["dev"])
    return [qa for qa in load_all() if qa.id in dev]
