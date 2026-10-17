#!/usr/bin/env python3
"""Diagnostics for neuro-symbolic narrowing (measurement only, no method applied).

1. ATA: can the small model name the tactic? Top-1 / top-2 accuracy over all 19
   tactic names in the snapshot (Enterprise, Mobile, ICS), and what re-ranking
   the top 20 by predicted tactic would do to recall@k.
2. VCA: oracle narrowing by other CWE fields (modes of introduction, applicable
   platforms) and re-ranking the top 20 by similarity to each CWE's CVE examples.

    uv run python scripts/diag_facets.py --model llama3:8b-instruct-fp16
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np

from ctiattr.candidates import CandidateGenerator, DenseIndex, rrf
from ctiattr.config import RUNS, EmbedConfig
from ctiattr.data import load_all
from ctiattr.retrieval import Embedder
from ctiattr.snapshot import Snapshot
from ctiattr.split import load_split

KS = (1, 3, 5, 20)

TACTIC_PROMPT = """Passage from a threat report:
"{passage}"

Which MITRE ATT&CK tactic (the adversary's goal) does the behavior in this passage serve?
Tactics: {tactics}

Return JSON with the two most likely tactics, most likely first."""


def ollama_json(base: str, model: str, prompt: str, schema: dict) -> dict:
    body = {"model": model, "stream": False, "format": schema, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": "You are a cyber threat intelligence analyst."},
                         {"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{base}/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.load(r)["message"]["content"])


def recall(rows, k):
    return sum(g in r[:k] for g, r in rows) / len(rows)


def line(name, rows, left=None):
    cells = "  ".join(f"@{k} {recall(rows, k):.2f}" for k in KS)
    extra = f"  | candidates left {left:.1f}" if left is not None else ""
    print(f"  {name:44}{cells}{extra}")


def cwe_facets(entry) -> dict[str, frozenset]:
    if entry.text.strip() in ("", "{}"):
        return {}
    c = json.loads(entry.text)
    as_list = lambda x: x if isinstance(x, list) else ([x] if x else [])
    phases = {i.get("Phase") for i in as_list((c.get("Modes_Of_Introduction") or {}).get("Introduction"))}
    plats = set()
    for kind, v in (c.get("Applicable_Platforms") or {}).items():
        for p in as_list(v):
            plats.add(f"{kind}:{p.get('@Name') or p.get('@Class')}")
    return {"phase": frozenset(p for p in phases if p), "platform": frozenset(plats)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b-instruct-fp16")
    ap.add_argument("--base", default="http://localhost:11434")
    args = ap.parse_args()

    emb = Embedder(EmbedConfig.from_env())
    s = Snapshot()
    gen = CandidateGenerator(emb, s)
    dev = set(load_split()["dev"])
    qs = [q for q in load_all() if q.id in dev]

    # ---- 1. ATA tactic classification ------------------------------------
    tactics = sorted({t for tid in s.technique_ids for t in s.tactics(tid)})
    schema = {"type": "object", "required": ["tactics"], "properties": {"tactics": {
        "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string", "enum": tactics}}}}
    allix = DenseIndex([e.doc_id for e in s.mitre], [f"{e.title}. {e.text}" for e in s.mitre], emb)
    cache_path = RUNS / f"diag_tactics__{args.model.replace(':', '-')}.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {json.loads(l)["id"]: json.loads(l) for l in cache_path.open()}

    base_rows, boost_rows, oracle_rows, rrf_rows = [], [], [], []
    top1 = top2 = 0
    ata = [q for q in qs if q.task == "ata"]
    for q in ata:
        if q.id not in cache:
            out = ollama_json(args.base, args.model,
                              TACTIC_PROMPT.format(passage=q.passage, tactics=", ".join(tactics)), schema)
            cache[q.id] = {"id": q.id, "pred": out["tactics"]}
            with cache_path.open("a") as f:
                f.write(json.dumps(cache[q.id]) + "\n")
        pred = cache[q.id]["pred"]
        gold = q.gold[0]
        gold_t = set(s.tactics(gold))
        top1 += pred[0] in gold_t
        top2 += bool(set(pred) & gold_t)
        r = rrf([[t for t, _ in allix.rank(q.question)], [t for t, _ in allix.rank(q.passage)]])[:20]
        level = lambda t: 2 if pred[0] in s.tactics(t) else 1 if pred[1] in s.tactics(t) else 0
        boosted = sorted(r, key=lambda t: -level(t))  # stable: keeps retrieval order within a level
        tactic_rank = [t for t in r if level(t) == 2] + [t for t in r if level(t) == 1]
        base_rows.append((gold, r))
        boost_rows.append((gold, boosted))
        rrf_rows.append((gold, rrf([r, tactic_rank])))
        oracle_rows.append((gold, [t for t in r if set(s.tactics(t)) & gold_t]))
    n = len(ata)
    print(f"\nATA tactic classification ({args.model}, n={n}, 19 tactic names)")
    print(f"  top-1 accuracy {top1 / n:.2f} ({top1}/{n}), top-2 accuracy {top2 / n:.2f} ({top2}/{n})")
    print("ATA top-20 re-ranking")
    line("as retrieved (all 1,076 techniques)", base_rows)
    line("predicted tactic first (soft, keeps all 20)", boost_rows)
    line("RRF(retrieval, predicted-tactic order)", rrf_rows)
    line("oracle tactic (upper bound)", oracle_rows, sum(len(r) for _, r in oracle_rows) / n)

    # ---- 2. VCA facets and CVE-example matching ---------------------------
    cwe = {e.doc_id.upper(): e for e in s.cwe}
    facets = {w: cwe_facets(e) for w, e in cwe.items()}
    by_cwe: dict[str, list[int]] = {}
    for i, c in enumerate(s.cve):
        for w in c.cwes:
            by_cwe.setdefault(w, []).append(i)
    rows = {k: [] for k in ("as retrieved", "oracle phase", "oracle platform", "oracle phase & platform",
                            "CVE-example max similarity first", "RRF(retrieval, CVE-example similarity)")}
    left = {k: 0.0 for k in rows}
    vca = [q for q in qs if q.task == "vca"]
    for q in vca:
        gold = q.gold[0]
        r = gen.vca(q.question, q.passage)["fused"][:20]
        g = facets.get(gold, {})

        def keep(w, keys):
            f = facets.get(w, {})
            return all(not g.get(k) or not f.get(k) or g[k] & f[k] for k in keys)

        variants = {"as retrieved": r,
                    "oracle phase": [w for w in r if keep(w, ["phase"])],
                    "oracle platform": [w for w in r if keep(w, ["platform"])],
                    "oracle phase & platform": [w for w in r if keep(w, ["phase", "platform"])]}
        sims = gen.cve_dense.scores(q.passage)
        best = {w: float(np.max(sims[by_cwe[w]])) for w in r if w in by_cwe}
        by_example = sorted(best, key=lambda w: -best[w]) + [w for w in r if w not in best]
        variants["CVE-example max similarity first"] = by_example
        variants["RRF(retrieval, CVE-example similarity)"] = rrf([r, by_example])
        for k, v in variants.items():
            rows[k].append((gold, v))
            left[k] += len(v)
    print(f"\nVCA top-20 narrowing (n={len(vca)}; a CWE with no value for a field is kept)")
    for k, v in rows.items():
        line(k, v, left[k] / len(vca) if k.startswith("oracle") else None)
    print(f"  CWEs in top 20 with CVE examples: "
          f"{sum(sum(w in by_cwe for w in r) for _, r in rows['as retrieved']) / len(vca):.1f} of 20 on average")


if __name__ == "__main__":
    main()
