"""Graph citations for an SGS answer: the snapshot paths that support the chosen ID.

Each citation is one path in the Neo4j graph, with the evidence text, its
similarity to the passage, the provenance of the edge that reaches the answer
(source, method, confidence) and a Cypher query that returns exactly that path.

ATA  report    ThreatReport -HAS_CHUNK-> ReportChunk -HAS_BEHAVIOR-> ReportBehavior -GROUNDS-> technique
               (the card's report examples and the behaviors that voted for it; never the
               question's own report), with the behavior's character span in its report
     capec     CAPEC -MAPS_TO_TECHNIQUE-> technique, for CAPEC patterns similar to the passage
     taxonomy  technique -SUBTECHNIQUE_OF-> parent, technique -IN_TACTIC-> tactic
VCA  capec     CAPEC -EXPLOITS_WEAKNESS-> CWE, for CAPEC patterns similar to the passage
     cve       CVE -HAS_WEAKNESS-> CWE, for CVE examples similar to the passage (NVD Primary/Secondary)

Citations go in the trace and the API response, never in the scored answer text
(the official scorer counts every ID it finds there).
"""

from __future__ import annotations

import json
from functools import cached_property

import numpy as np

from ctiattr import provenance as pv
from ctiattr.config import ROOT
from ctiattr.schemas import Citation, PathStep

REPORT_VOTE = 30     # behaviors in the ATA report vote (sgs.ata_candidates)
CAPEC_TOP = 10       # CAPEC hits aggregated by the bridges (candidates.py)
CVE_TOP = 20         # CVE hits aggregated by the CVE vote
PER_KIND = 3         # citations kept per kind, best first


def cypher_for(path: list[PathStep]) -> str:
    """A MATCH that returns exactly this path (Tactic nodes are keyed by name, all others by id)."""
    parts = []
    for i, step in enumerate(path):
        if step.node:
            key = "name" if step.label == "Tactic" else "id"
            parts.append(f"(:{step.label} {{{key}: '{step.node}'}})")
        else:
            parts.append(f"-[:{step.rel}]->")
    return f"MATCH p = {''.join(parts)} RETURN p"


def _path(*steps) -> list[PathStep]:
    """Alternating (label, id) nodes and relationship names."""
    return [PathStep(rel=s) if isinstance(s, str) else PathStep(label=s[0], node=s[1]) for s in steps]


class Citer:
    def __init__(self, sgs):
        self.sgs = sgs
        self.s = sgs.s
        self.capec = {c.id: c for c in self.s.capec}
        self.cve = {c.id: c for c in self.s.cve}
        self.chunks = {c["chunk_id"]: c for c in map(json.loads, (ROOT / "data" / "derived" /
                                                                 "report_chunks.jsonl").open())}

    @cached_property
    def grounds(self) -> list[pv.ExtractionSignals]:
        """Tier and confidence inputs for every report-layer behavior (same order as sgs.ttps)."""
        return pv.extraction_signals(self.s, self.sgs.emb, self.sgs.ttps, self.chunks)

    @cached_property
    def tier_confidence(self) -> dict[str, float]:
        return {t: v["precision"] for t, v in pv.audit_precision().items()}

    def cite(self, qa, chosen: str, shown_card: str = "") -> list[Citation]:
        return self._ata(qa, chosen, shown_card) if qa.task == "ata" else self._vca(qa, chosen, shown_card)

    # --- ATA ---------------------------------------------------------------

    def _ata(self, qa, tid: str, card: str) -> list[Citation]:
        sgs, out = self.sgs, []
        pvec = sgs.emb.embed_one(qa.passage)
        sims = sgs.behaviors @ pvec
        others = [i for i in np.argsort(-sims) if sgs.ttps[i]["report_id"] != qa.report_id]
        voters = [i for i in others[:REPORT_VOTE] if sgs.ttps[i]["technique_id"] == tid]
        on_card = sorted((i for i in sgs.examples_of.get(tid, []) if sgs.ttps[i]["report_id"] != qa.report_id),
                         key=lambda i: -sims[i])[:2]
        for i in list(dict.fromkeys(on_card + voters))[:PER_KIND]:
            b, sig = sgs.ttps[i], self.grounds[i]
            path = _path(("ThreatReport", b["report_id"]), "HAS_CHUNK", ("ReportChunk", b["chunk_id"]),
                         "HAS_BEHAVIOR", ("ReportBehavior", f"{b['chunk_id']}@{i}"), "GROUNDS",
                         ("AttackTechnique", tid))
            out.append(Citation(
                kind="report", path=path, cypher=cypher_for(path), evidence=b["behavior"],
                similarity=round(float(sims[i]), 4), shown_to_model=i in on_card, in_report_vote=i in voters,
                source="report_layer", method="extracted", confidence=self.tier_confidence.get(sig.tier),
                tier=sig.tier, span=[b["report_id"], sig.start, sig.end]))

        hits = sgs.gen.capec_dense.rank(qa.passage)[:CAPEC_TOP]
        for cid, sim in [(c, s) for c, s in hits if tid in self.capec[c].techniques][:PER_KIND]:
            path = _path(("CAPEC", cid), "MAPS_TO_TECHNIQUE", ("AttackTechnique", tid))
            out.append(Citation(kind="capec", path=path, cypher=cypher_for(path), evidence=self.capec[cid].name,
                                similarity=round(sim, 4), source="cticonnect:corpus_kb", method="asserted",
                                confidence=1.0))

        parent = self.s.parent(tid)
        if parent in self.s.technique_ids:
            path = _path(("AttackTechnique", tid), "SUBTECHNIQUE_OF", ("AttackTechnique", parent))
            out.append(Citation(kind="taxonomy", path=path, cypher=cypher_for(path),
                                evidence=sgs.title.get(parent, ""), source="cticonnect:corpus_kb",
                                method="derived", confidence=1.0, shown_to_model=False))
        for tactic in self.s.tactics(tid):
            path = _path(("AttackTechnique", tid), "IN_TACTIC", ("Tactic", tactic))
            out.append(Citation(kind="taxonomy", path=path, cypher=cypher_for(path), evidence=tactic,
                                source="cticonnect:corpus_kb", method="asserted", confidence=1.0,
                                shown_to_model=tactic in card))
        return out

    # --- VCA ---------------------------------------------------------------

    def _vca(self, qa, wid: str, card: str) -> list[Citation]:
        out = []
        hits = self.sgs.gen.capec_dense.rank(qa.passage)[:CAPEC_TOP]
        for cid, sim in [(c, s) for c, s in hits if wid in self.capec[c].cwes][:PER_KIND]:
            path = _path(("CAPEC", cid), "EXPLOITS_WEAKNESS", ("CWE", wid))
            out.append(Citation(kind="capec", path=path, cypher=cypher_for(path), evidence=self.capec[cid].name,
                                similarity=round(sim, 4), shown_to_model=self.capec[cid].name in card,
                                in_report_vote=False, source="cticonnect:corpus_kb", method="asserted",
                                confidence=1.0))
        hits = self.sgs.gen.cve_dense.rank(qa.passage)[:CVE_TOP]
        for vid, sim in [(v, s) for v, s in hits if wid in self.cve[v].cwes][:PER_KIND]:
            c = self.cve[vid]
            path = _path(("CVE", vid), "HAS_WEAKNESS", ("CWE", wid))
            out.append(Citation(kind="cve", path=path, cypher=cypher_for(path), evidence=c.description[:220],
                                similarity=round(sim, 4), shown_to_model=c.description[:220] in card,
                                source="cticonnect:corpus_kb", method="asserted", confidence=1.0,
                                weakness_type="Primary" if wid in c.primary else "Secondary"))
        return out
