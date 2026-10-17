#!/usr/bin/env python3
"""ATA selection with report-layer examples on the evidence cards (measurement only).

Two configurations, each scored with listwise (retrieval order), a 5-rotation
majority vote and the pairwise tournament over the top 5:
  old-candidates  RRF(question, passage) over all techniques  + cards with examples
  new-candidates  RRF(question, passage, report-example vote) + cards with examples
Examples come from data/derived/report_ttps.jsonl and never from the question's
own source report. Prompts and the call cache are shared with diag_selection.py.

    uv run python scripts/diag_ata_examples.py --model llama3:8b-instruct-fp16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_selection import LISTWISE, PAIRWISE, TAXONOMY, Cards, Ollama  # noqa: E402

from ctiattr.candidates import CandidateGenerator, DenseIndex, rrf  # noqa: E402
from ctiattr.config import ROOT, RUNS, EmbedConfig  # noqa: E402
from ctiattr.data import load_all  # noqa: E402
from ctiattr.retrieval import Embedder  # noqa: E402
from ctiattr.snapshot import Snapshot  # noqa: E402
from ctiattr.split import load_split  # noqa: E402

DERIVED = ROOT / "data" / "derived"


def sentence_with(chunk: str, phrase: str, limit: int = 160) -> str:
    """The sentence containing the phrase, cut to `limit` characters centred on the phrase."""
    for sent in re.split(r"(?<=[.!?])\s+", chunk):
        sent = re.sub(r"\s+", " ", sent)
        at = sent.find(phrase)
        if at >= 0:
            lo = max(0, min(at - (limit - len(phrase)) // 2, len(sent) - limit))
            return ("…" if lo else "") + sent[lo:lo + limit] + ("…" if lo + limit < len(sent) else "")
    return phrase


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b-instruct-fp16")
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    emb = Embedder(EmbedConfig.from_env())
    s = Snapshot()
    cards = Cards(s, CandidateGenerator(emb, s))
    llm = Ollama(args.base, args.model, RUNS / f"diag_selection__{args.model.replace(':', '-')}.jsonl")
    allix = DenseIndex([e.doc_id for e in s.mitre], [f"{e.title}. {e.text}" for e in s.mitre], emb)
    ttps = [json.loads(l) for l in (DERIVED / "report_ttps.jsonl").open()]
    chunks = {json.loads(l)["chunk_id"]: json.loads(l)["text"] for l in (DERIVED / "report_chunks.jsonl").open()}
    B = emb.embed([b["behavior"] for b in ttps])
    by_tech = defaultdict(list)
    for i, b in enumerate(ttps):
        by_tech[b["technique_id"]].append(i)

    dev = set(load_split()["dev"])
    ata = [q for q in load_all() if q.task == "ata" and q.id in dev]

    prep = {}
    for q in ata:
        sims = B @ emb.embed_one(q.passage)
        allowed = lambda i: ttps[i]["report_id"] != q.report_id
        nearest = [i for i in np.argsort(-sims) if allowed(i)][:30]
        score = defaultdict(float)
        for i in nearest:
            score[ttps[i]["technique_id"]] += float(sims[i])
        vote = sorted(score, key=lambda t: -score[t])
        base = rrf([[t for t, _ in allix.rank(q.question)], [t for t, _ in allix.rank(q.passage)]])

        def card(t):
            ex = sorted((i for i in by_tech.get(t, []) if allowed(i)), key=lambda i: -sims[i])[:2]
            shown = "; ".join(f'"{sentence_with(chunks[ttps[i]["chunk_id"]], ttps[i]["behavior"])}"' for i in ex)
            return cards.ata(t) + (f" | report examples: {shown}" if shown else " | report examples: none")

        prep[q.id] = {}
        for cfg, ranked in (("old", base[:5]), ("new", rrf([base, vote])[:5])):
            prep[q.id][cfg] = (ranked, {t: card(t) for t in ranked})

    def listwise(q, cfg, order, key):
        _, c = prep[q.id][cfg]
        schema = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string", "enum": order}}}
        return llm(key, LISTWISE.format(taxonomy=TAXONOMY["ata"], passage=q.passage,
                                        cards="\n".join(c[i] for i in order)), schema)["id"]

    def pairwise(q, cfg):
        ranked, c = prep[q.id][cfg]
        schema = {"type": "object", "required": ["choice"],
                  "properties": {"choice": {"type": "string", "enum": ["A", "B"]}}}
        leader = ranked[0]
        for ch in ranked[1:]:
            ask = lambda a, b, tag: llm(f"expair:{cfg}:{q.id}:{a}:{b}:{tag}",
                                        PAIRWISE.format(taxonomy=TAXONOMY["ata"], passage=q.passage,
                                                        a=c[a], b=c[b]), schema)["choice"]
            if ask(leader, ch, "ab") == "B" and ask(ch, leader, "ba") == "A":
                leader = ch
        return leader

    def run(q):
        out = {}
        for cfg in ("old", "new"):
            ranked, _ = prep[q.id][cfg]
            picks = [listwise(q, cfg, ranked[k:] + ranked[:k], f"exlist:{cfg}:{k}:{q.id}") for k in range(5)]
            votes = Counter(picks)
            top = max(votes.values())
            out[cfg] = {"listwise (retrieval order)": picks[0],
                        "5-rotation vote": min((p for p in votes if votes[p] == top), key=ranked.index),
                        "pairwise tournament": pairwise(q, cfg), "agree5": top == 5}
        return q.id, out

    with ThreadPoolExecutor(args.concurrency) as ex:
        res = dict(ex.map(run, ata))

    n = len(ata)
    print(f"ATA selection with report examples on cards (n={n})")
    print("  previous (no examples, old candidates): listwise 0.33, rotation vote 0.27, pairwise 0.40")
    for cfg, label in (("old", "old candidates + example cards"), ("new", "new candidates + example cards")):
        shown = sum(q.gold[0] in prep[q.id][cfg][0] for q in ata)
        print(f"\n  {label} (gold in top 5: {shown}/{n})")
        for m in ("listwise (retrieval order)", "5-rotation vote", "pairwise tournament"):
            acc = sum(res[q.id][cfg][m] == q.gold[0] for q in ata)
            print(f"    {m:28} accuracy {acc / n:.2f} ({acc}/{n}) | when shown {acc}/{shown} = {acc / max(shown, 1):.2f}")
        print(f"    all 5 rotations agree on {sum(res[q.id][cfg]['agree5'] for q in ata)}/{n} questions")


if __name__ == "__main__":
    main()
