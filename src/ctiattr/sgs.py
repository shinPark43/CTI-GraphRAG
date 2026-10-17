"""Snapshot-Guided Selection (SGS): the frozen attribution method.

Everything the model sees comes from the CTIConnect snapshot (corpus_kb plus the
report layer built from corpus_reports); the model only chooses among listed IDs.

ATA  candidates: RRF of dense(question), dense(passage) over all 1,076 techniques
                 and a vote over the 30 report-layer behaviors nearest the passage
     cards:      title, tactics, platforms, detection advice, 2 report examples
     choice:     one listwise call over the top 5, in retrieval order
VCA  candidates: RRF of dense(question) over CWE records, the CAPEC bridge and
                 the CVE-example vote
     cards:      title, consequences, a related CAPEC pattern, the closest CVE example
     choice:     5 listwise calls over cyclic rotations of the top 5, majority vote
                 (ties -> better retrieval rank)

Report-layer examples never come from the question's own source report. The
answer text holds exactly one ID (the official scorer counts every ID it finds).
Configuration frozen on the dev split (scripts/diag_*.py).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np

from ctiattr.candidates import CandidateGenerator, DenseIndex, rrf
from ctiattr.config import ROOT
from ctiattr.data import QA
from ctiattr.retrieval import Embedder
from ctiattr.snapshot import Snapshot
from ctiattr.vendor.cticonnect_metrics import extract_ids

TOP = 5
SYSTEM = "You are a cyber threat intelligence analyst."
TAXONOMY = {"ata": "MITRE ATT&CK techniques", "vca": "CWE weaknesses"}
KIND = {"ata": "mitre", "vca": "cwe"}

LISTWISE = """You are mapping a threat-report passage to {taxonomy}.

Passage:
"{passage}"

Candidates:
{cards}

Choose the ONE candidate that best describes the behavior in the passage. Judge by
what each candidate means, not by shared words. Return JSON with its id."""


def _as_list(x):
    return x if isinstance(x, list) else ([x] if x else [])


def _sentence_with(chunk: str, phrase: str, limit: int = 160) -> str:
    """The sentence containing the phrase, cut to `limit` characters centred on the phrase."""
    for sent in re.split(r"(?<=[.!?])\s+", chunk):
        sent = re.sub(r"\s+", " ", sent)
        at = sent.find(phrase)
        if at >= 0:
            lo = max(0, min(at - (limit - len(phrase)) // 2, len(sent) - limit))
            return ("…" if lo else "") + sent[lo:lo + limit] + ("…" if lo + limit < len(sent) else "")
    return phrase


class SGS:
    def __init__(self, embedder: Embedder, snapshot: Snapshot | None = None):
        self.emb = embedder
        self.s = s = snapshot or Snapshot()
        self.gen = CandidateGenerator(embedder, s)
        self.all_tech = DenseIndex([e.doc_id for e in s.mitre], [f"{e.title}. {e.text}" for e in s.mitre],
                                   embedder)
        self.title = {e.doc_id.upper(): e.title for e in s.mitre} | {e.doc_id.upper(): e.title for e in s.cwe}

        derived = ROOT / "data" / "derived"
        self.ttps = [json.loads(l) for l in (derived / "report_ttps.jsonl").open()]
        self.chunks = {json.loads(l)["chunk_id"]: json.loads(l)["text"]
                       for l in (derived / "report_chunks.jsonl").open()}
        self.behaviors = embedder.embed([b["behavior"] for b in self.ttps])
        self.examples_of = defaultdict(list)
        for i, b in enumerate(self.ttps):
            self.examples_of[b["technique_id"]].append(i)

        self.cwe = {e.doc_id.upper(): e for e in s.cwe}
        self.capec_for = defaultdict(list)
        for c in s.capec:
            for w in c.cwes:
                self.capec_for[w].append(c.name)
        self.cves_for = defaultdict(list)
        for i, c in enumerate(s.cve):
            for w in c.cwes:
                self.cves_for[w].append(i)

    # --- ATA ---------------------------------------------------------------

    def _ata_card(self, tid: str, sims: np.ndarray, report_id: str) -> str:
        e = self.s._mitre_by_id[tid]
        c = json.loads(e.text)
        det = (c.get("x_mitre_detection") or "").split(". ")[0][:220]
        card = (f"{tid} | {e.title} | tactics: {', '.join(self.s.tactics(tid))} | "
                f"platforms: {', '.join((c.get('x_mitre_platforms') or [])[:4])} | detection: {det}")
        ex = sorted((i for i in self.examples_of.get(tid, []) if self.ttps[i]["report_id"] != report_id),
                    key=lambda i: -sims[i])[:2]
        shown = "; ".join(f'"{_sentence_with(self.chunks[self.ttps[i]["chunk_id"]], self.ttps[i]["behavior"])}"'
                          for i in ex)
        return card + (f" | report examples: {shown}" if shown else " | report examples: none")

    def ata_candidates(self, qa: QA) -> tuple[list[str], dict[str, str]]:
        sims = self.behaviors @ self.emb.embed_one(qa.passage)
        nearest = [i for i in np.argsort(-sims) if self.ttps[i]["report_id"] != qa.report_id][:30]
        score = defaultdict(float)
        for i in nearest:
            score[self.ttps[i]["technique_id"]] += float(sims[i])
        vote = sorted(score, key=lambda t: -score[t])
        base = rrf([[t for t, _ in self.all_tech.rank(qa.question)],
                    [t for t, _ in self.all_tech.rank(qa.passage)]])
        ranked = rrf([base, vote])[:TOP]
        return ranked, {t: self._ata_card(t, sims, qa.report_id) for t in ranked}

    # --- VCA ---------------------------------------------------------------

    def _vca_card(self, wid: str, cve_sims: np.ndarray) -> str:
        e = self.cwe[wid]
        parts = [f"{wid} | {e.title}"]
        if e.text.strip() not in ("", "{}"):
            c = json.loads(e.text)
            impacts = []
            for x in _as_list((c.get("Common_Consequences") or {}).get("Consequence"))[:2]:
                impacts += [str(i) for i in _as_list(x.get("Impact"))][:2]
            if impacts:
                parts.append("consequences: " + "; ".join(impacts))
        if wid in self.capec_for:
            parts.append(f"related attack pattern: {self.capec_for[wid][0]}")
        if wid in self.cves_for:
            best = max(self.cves_for[wid], key=lambda i: cve_sims[i])
            parts.append(f"example vulnerability: {self.s.cve[best].description[:220]}")
        return " | ".join(parts)

    def vca_candidates(self, qa: QA) -> tuple[list[str], dict[str, str]]:
        ranked = self.gen.vca(qa.question, qa.passage)["fused"][:TOP]
        cve_sims = self.gen.cve_dense.scores(qa.passage)
        return ranked, {w: self._vca_card(w, cve_sims) for w in ranked}

    # --- choice ------------------------------------------------------------

    def _pick(self, llm, qa: QA, order: list[str], cards: dict[str, str]):
        schema = {"type": "object", "required": ["id"], "additionalProperties": False,
                  "properties": {"id": {"type": "string", "enum": order}}}
        prompt = LISTWISE.format(taxonomy=TAXONOMY[qa.task], passage=qa.passage,
                                 cards="\n".join(cards[i] for i in order))
        out, call = llm.chat_json(prompt, schema, system=SYSTEM)
        return out["id"], call

    def answer(self, llm, qa: QA) -> tuple[str, dict]:
        if qa.task == "ata":
            ranked, cards = self.ata_candidates(qa)
            rotations = 1
        else:
            ranked, cards = self.vca_candidates(qa)
            rotations = TOP
        picks, calls = [], []
        for k in range(rotations):
            pick, call = self._pick(llm, qa, ranked[k:] + ranked[:k], cards)
            picks.append(pick)
            calls.append(call.to_dict())
        votes = Counter(picks)
        top = max(votes.values())
        chosen = min((p for p in votes if votes[p] == top), key=ranked.index)

        # Final check: exactly one ID in the answer text, and it is a listed snapshot ID.
        assert chosen in ranked
        title = self.title.get(chosen.upper(), "")
        if extract_ids(title, KIND[qa.task]):
            title = ""
        text = f"{chosen} — {title}" if title else chosen
        return text, {"candidates": ranked, "cards": cards, "picks": picks, "chosen": chosen,
                      "retrieved": [ranked], "calls": calls}
