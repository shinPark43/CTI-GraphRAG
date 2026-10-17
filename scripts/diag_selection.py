#!/usr/bin/env python3
"""Diagnostics for the final decision (measurement only, no method applied).

A. Selection accuracy: given the top-K retrieved candidates as snapshot evidence
   cards, how often does the small model pick the gold ID?
     - listwise, K = 5 and K = 10 (one call, JSON enum of the candidate IDs)
     - pairwise "king of the hill" over the top 5: leader vs next candidate, asked
       in both orders; a challenger wins only if both orders agree
B. Tactic classification again, with each tactic described by the titles of its
   techniques in the snapshot (no outside text).

All candidates come from all 1,076 techniques (no domain filter). Every model
call is cached in runs/diag_selection__<model>.jsonl.

    uv run python scripts/diag_selection.py --model llama3:8b-instruct-fp16
"""

from __future__ import annotations

import argparse
import json
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ctiattr.candidates import CandidateGenerator, DenseIndex, rrf
from ctiattr.config import RUNS, EmbedConfig
from ctiattr.data import load_all
from ctiattr.retrieval import Embedder
from ctiattr.snapshot import Snapshot
from ctiattr.split import load_split

TAXONOMY = {"ata": "MITRE ATT&CK techniques", "vca": "CWE weaknesses"}

LISTWISE = """You are mapping a threat-report passage to {taxonomy}.

Passage:
"{passage}"

Candidates:
{cards}

Choose the ONE candidate that best describes the behavior in the passage. Judge by
what each candidate means, not by shared words. Return JSON with its id."""

PAIRWISE = """You are mapping a threat-report passage to {taxonomy}.

Passage:
"{passage}"

Candidate A: {a}

Candidate B: {b}

Which candidate better describes the behavior in the passage? Return JSON with "A" or "B"."""

TACTIC = """Passage from a threat report:
"{passage}"

Which MITRE ATT&CK tactic (the adversary's goal) does the behavior in this passage serve?
Each tactic is listed with techniques that belong to it:
{definitions}

Return JSON with the two most likely tactics, most likely first."""


class Ollama:
    def __init__(self, base: str, model: str, cache_path):
        self.base, self.model, self.path = base, model, cache_path
        self.lock = threading.Lock()
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.open():
                r = json.loads(line)
                self.cache[r["key"]] = r["out"]

    def __call__(self, key: str, prompt: str, schema: dict) -> dict:
        if key in self.cache:
            return self.cache[key]
        body = {"model": self.model, "stream": False, "format": schema, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": "You are a cyber threat intelligence analyst."},
                             {"role": "user", "content": prompt}]}
        req = urllib.request.Request(f"{self.base}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(json.load(r)["message"]["content"])
        with self.lock:
            self.cache[key] = out
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "out": out}) + "\n")
        return out


def _as_list(x):
    return x if isinstance(x, list) else ([x] if x else [])


class Cards:
    """One-line evidence cards built only from snapshot fields."""

    def __init__(self, s: Snapshot, gen: CandidateGenerator):
        self.s, self.gen = s, gen
        self.cwe = {e.doc_id.upper(): e for e in s.cwe}
        self.capec_for = {}
        for c in s.capec:
            for w in c.cwes:
                self.capec_for.setdefault(w, []).append(c.name)
        self.cves_for = {}
        for i, c in enumerate(s.cve):
            for w in c.cwes:
                self.cves_for.setdefault(w, []).append(i)

    def ata(self, tid: str) -> str:
        e = self.s._mitre_by_id[tid]
        c = json.loads(e.text)
        det = (c.get("x_mitre_detection") or "").split(". ")[0][:220]
        return (f"{tid} | {e.title} | tactics: {', '.join(self.s.tactics(tid))} | "
                f"platforms: {', '.join((c.get('x_mitre_platforms') or [])[:4])} | detection: {det}")

    def vca(self, wid: str, cve_sims: np.ndarray) -> str:
        e = self.cwe[wid]
        parts = [f"{wid} | {e.title}"]
        if e.text.strip() not in ("", "{}"):
            c = json.loads(e.text)
            cons = _as_list((c.get("Common_Consequences") or {}).get("Consequence"))
            impacts = []
            for x in cons[:2]:
                impacts += [str(i) for i in _as_list(x.get("Impact"))][:2]
            if impacts:
                parts.append("consequences: " + "; ".join(impacts))
        if wid in self.capec_for:
            parts.append(f"related attack pattern: {self.capec_for[wid][0]}")
        if wid in self.cves_for:
            best = max(self.cves_for[wid], key=lambda i: cve_sims[i])
            parts.append(f"example vulnerability: {self.s.cve[best].description[:220]}")
        return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b-instruct-fp16")
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    emb = Embedder(EmbedConfig.from_env())
    s = Snapshot()
    gen = CandidateGenerator(emb, s)
    cards = Cards(s, gen)
    llm = Ollama(args.base, args.model, RUNS / f"diag_selection__{args.model.replace(':', '-')}.jsonl")
    allix = DenseIndex([e.doc_id for e in s.mitre], [f"{e.title}. {e.text}" for e in s.mitre], emb)
    dev = set(load_split()["dev"])
    qs = [q for q in load_all() if q.id in dev]

    # candidates + cards per question
    prep = {}
    for q in qs:
        if q.task == "ata":
            ranked = rrf([[t for t, _ in allix.rank(q.question)], [t for t, _ in allix.rank(q.passage)]])[:10]
            card = {t: cards.ata(t) for t in ranked}
        else:
            ranked = gen.vca(q.question, q.passage)["fused"][:10]
            sims = gen.cve_dense.scores(q.passage)
            card = {w: cards.vca(w, sims) for w in ranked}
        prep[q.id] = (ranked, card)

    def listwise(q, k):
        ranked, card = prep[q.id]
        ids = ranked[:k]
        schema = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string", "enum": ids}}}
        prompt = LISTWISE.format(taxonomy=TAXONOMY[q.task], passage=q.passage,
                                 cards="\n".join(card[i] for i in ids))
        return llm(f"list{k}:{q.id}", prompt, schema)["id"]

    def pairwise(q):
        ranked, card = prep[q.id]
        schema = {"type": "object", "required": ["choice"],
                  "properties": {"choice": {"type": "string", "enum": ["A", "B"]}}}
        leader = ranked[0]
        for challenger in ranked[1:5]:
            ask = lambda a, b, tag: llm(f"pair:{q.id}:{a}:{b}:{tag}",
                                        PAIRWISE.format(taxonomy=TAXONOMY[q.task], passage=q.passage,
                                                        a=card[a], b=card[b]), schema)["choice"]
            first = ask(leader, challenger, "ab")       # leader shown as A
            second = ask(challenger, leader, "ba")      # leader shown as B
            if first == "B" and second == "A":
                leader = challenger
        return leader

    with ThreadPoolExecutor(args.concurrency) as ex:
        picks = {name: dict(zip([q.id for q in qs], ex.map(fn, qs))) for name, fn in [
            ("listwise top-5", lambda q: listwise(q, 5)),
            ("listwise top-10", lambda q: listwise(q, 10)),
            ("pairwise top-5", pairwise),
        ]}

    for task in ("ata", "vca"):
        tq = [q for q in qs if q.task == task]
        n = len(tq)
        print(f"\n{task.upper()} selection (n={n}){'':6}accuracy   | when gold is among the shown candidates")
        rows = [("retrieval top-1, no LLM", {q.id: prep[q.id][0][0] for q in tq}, 1)]
        rows += [(name, p, 10 if "10" in name else 5) for name, p in picks.items()]
        for name, p, k in rows:
            acc = sum(p[q.id] == q.gold[0] for q in tq)
            shown = [q for q in tq if q.gold[0] in prep[q.id][0][:k]]
            cond = sum(p[q.id] == q.gold[0] for q in shown)
            print(f"  {name:28} {acc / n:.2f} ({acc}/{n})   | {cond}/{len(shown)}"
                  f" = {cond / max(len(shown), 1):.2f}")
        for name, p in picks.items():
            pos = Counter(prep[q.id][0].index(p[q.id]) + 1 for q in tq)
            print(f"    {name}: picked rank {dict(sorted(pos.items()))}")

    # ---- B. tactic classification with snapshot-derived definitions -------
    tactics = sorted({t for tid in s.technique_ids for t in s.tactics(tid)})
    titles = {t: [] for t in tactics}
    for e in sorted(s.mitre, key=lambda e: e.doc_id):
        if "." not in e.doc_id:
            for t in s.tactics(e.doc_id):
                if e.title not in titles[t]:
                    titles[t].append(e.title)
    defs = "\n".join(f"- {t}: {', '.join(titles[t][:15])}" for t in tactics)
    schema = {"type": "object", "required": ["tactics"], "properties": {"tactics": {
        "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string", "enum": tactics}}}}
    ata = [q for q in qs if q.task == "ata"]
    with ThreadPoolExecutor(args.concurrency) as ex:
        preds = list(ex.map(lambda q: llm(f"tacticdef:{q.id}", TACTIC.format(passage=q.passage, definitions=defs),
                                          schema)["tactics"], ata))
    top1 = sum(p[0] in s.tactics(q.gold[0]) for q, p in zip(ata, preds))
    top2 = sum(bool(set(p) & set(s.tactics(q.gold[0]))) for q, p in zip(ata, preds))
    print(f"\nATA tactic with snapshot-derived definitions: top-1 {top1}/{len(ata)} = {top1 / len(ata):.2f}, "
          f"top-2 {top2}/{len(ata)} = {top2 / len(ata):.2f}   (names only: 0.50 / 0.63)")
    rows = []
    for q, p in zip(ata, preds):
        r = rrf([[t for t, _ in allix.rank(q.question)], [t for t, _ in allix.rank(q.passage)]])[:20]
        level = lambda t: 2 if p[0] in s.tactics(t) else 1 if p[1] in s.tactics(t) else 0
        rows.append((q.gold[0], sorted(r, key=lambda t: -level(t))))
    print("  re-ranked top 20, predicted tactic first: " +
          "  ".join(f"@{k} {sum(g in r[:k] for g, r in rows) / len(rows):.2f}" for k in (1, 3, 5)) +
          "   (as retrieved: @1 0.33  @3 0.43  @5 0.57)")


if __name__ == "__main__":
    main()
