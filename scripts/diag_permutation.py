#!/usr/bin/env python3
"""Position-bias check for listwise selection (measurement only).

The top-5 candidates are shown in 5 cyclic rotations, so each candidate appears
once in every position; the answer is the majority vote (ties -> better retrieval
rank). Also: one call with the order reversed. Reuses the evidence cards, prompt
and model-call cache of diag_selection.py (rotation 0 = its listwise top-5).

    uv run python scripts/diag_permutation.py --model llama3:8b-instruct-fp16
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_selection import LISTWISE, TAXONOMY, Cards, Ollama  # noqa: E402

from ctiattr.candidates import CandidateGenerator, DenseIndex, rrf  # noqa: E402
from ctiattr.config import RUNS, EmbedConfig  # noqa: E402
from ctiattr.data import load_all  # noqa: E402
from ctiattr.retrieval import Embedder  # noqa: E402
from ctiattr.snapshot import Snapshot  # noqa: E402
from ctiattr.split import load_split  # noqa: E402


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

    prep = {}
    for q in qs:
        if q.task == "ata":
            ranked = rrf([[t for t, _ in allix.rank(q.question)], [t for t, _ in allix.rank(q.passage)]])[:5]
            prep[q.id] = (ranked, {t: cards.ata(t) for t in ranked})
        else:
            ranked = gen.vca(q.question, q.passage)["fused"][:5]
            sims = gen.cve_dense.scores(q.passage)
            prep[q.id] = (ranked, {w: cards.vca(w, sims) for w in ranked})

    def ask(q, order, key):
        _, card = prep[q.id]
        schema = {"type": "object", "required": ["id"], "properties": {"id": {"type": "string", "enum": order}}}
        prompt = LISTWISE.format(taxonomy=TAXONOMY[q.task], passage=q.passage,
                                 cards="\n".join(card[i] for i in order))
        return llm(key, prompt, schema)["id"]

    def run(q):
        ranked, _ = prep[q.id]
        picks = []
        for k in range(5):
            order = ranked[k:] + ranked[:k]
            picks.append(ask(q, order, f"list5:{q.id}" if k == 0 else f"list5rot{k}:{q.id}"))
        reverse = ask(q, ranked[::-1], f"list5rev:{q.id}")
        votes = Counter(picks)
        best = max(votes.values())
        vote = min((c for c in votes if votes[c] == best), key=ranked.index)
        return q.id, {"original": picks[0], "reversed": reverse, "rotation vote": vote,
                      "picked position": [(ranked[k:] + ranked[:k]).index(p) + 1 for k, p in enumerate(picks)],
                      "agreement": best}

    with ThreadPoolExecutor(args.concurrency) as ex:
        res = dict(ex.map(run, qs))

    for task in ("ata", "vca"):
        tq = [q for q in qs if q.task == task]
        shown = [q for q in tq if q.gold[0] in prep[q.id][0]]
        print(f"\n{task.upper()} listwise top-5 (n={len(tq)}, gold shown in {len(shown)})")
        for name in ("original", "reversed", "rotation vote"):
            acc = sum(res[q.id][name] == q.gold[0] for q in tq)
            print(f"  {name:16} accuracy {acc / len(tq):.2f} ({acc}/{len(tq)})  | when shown {acc}/{len(shown)}"
                  f" = {acc / max(len(shown), 1):.2f}")
        pos = Counter(p for q in tq for p in res[q.id]["picked position"])
        print(f"  position picked across all rotations (1 = shown first): {dict(sorted(pos.items()))}")
        print(f"  questions where all 5 rotations agree: {sum(res[q.id]['agreement'] == 5 for q in tq)}/{len(tq)}")


if __name__ == "__main__":
    main()
