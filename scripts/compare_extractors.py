#!/usr/bin/env python3
"""Schema-constrained extraction with the local model vs the committed report layer.

Runs the report-layer extractor (same CTIFoundry prompt, same ExtractionReply JSON
schema, Pydantic validation) with a local Ollama model on a fixed sample of chunks
and compares it with the committed gpt-5.4-mini layer on the same chunks:

  - schema / unknown-ID / not-verbatim drop rates, records kept, latency per chunk
  - agreement on (chunk, technique) pairs, exact ID and technique family (T1059.001 -> T1059)

    uv run python scripts/compare_extractors.py --chunks 40          # -> runs/compare_extractors.json
    uv run python scripts/compare_extractors.py --smoke-openai        # one strict-schema call to gpt-5.4-mini

Replies are cached in .cache/report_ttps_raw__<model>.jsonl, shared with build_report_layer.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_report_layer import DEFAULT_MODEL, extraction_cache  # noqa: E402

from ctiattr.config import CACHE, ROOT, RUNS, EmbedConfig, LLMConfig  # noqa: E402
from ctiattr.report_layer import Chunk, extract, validate  # noqa: E402
from ctiattr.snapshot import Snapshot  # noqa: E402

DERIVED = ROOT / "data" / "derived"


def pairs(records, family=False):
    f = (lambda t: t.split(".")[0]) if family else (lambda t: t)
    return {(r["chunk_id"], f(r["technique_id"])) for r in records}


def agreement(mine: set, ref: set) -> dict:
    both = len(mine & ref)
    p = both / len(mine) if mine else 0.0
    r = both / len(ref) if ref else 0.0
    return {"precision_vs_ref": round(p, 3), "recall_vs_ref": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "shared": both,
            "local": len(mine), "ref": len(ref)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b-instruct-fp16")
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--smoke-openai", action="store_true", help="only check that OpenAI accepts the strict schema")
    args = ap.parse_args()

    s = Snapshot()
    tactics = sorted({t for tid in s.technique_ids for t in s.tactics(tid)})
    chunks = [Chunk(**json.loads(l)) for l in (DERIVED / "report_chunks.jsonl").open()]

    if args.smoke_openai:
        c = chunks[0]
        raw = extract(OpenAI(api_key=EmbedConfig.from_env().api_key), DEFAULT_MODEL, c, tactics)
        kept, why = validate(c, raw, s.technique_ids)
        print(f"{DEFAULT_MODEL} strict schema accepted: {len(raw)} records, kept {len(kept)}, dropped {why}")
        return

    sample = sorted(random.Random(6338).sample(chunks, args.chunks), key=lambda c: c.chunk_id)
    cfg = LLMConfig.from_env(args.model)
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=900)
    CACHE.mkdir(exist_ok=True)
    path = extraction_cache(args.model)
    cache = {}
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            cache[r["chunk_id"]] = r
    kept, why, raw_n, lat = [], Counter(), 0, []
    for i, c in enumerate(sample, 1):
        if c.chunk_id not in cache:
            t0 = time.perf_counter()
            raw = extract(client, args.model, c, tactics, args.temperature)
            rec = {"chunk_id": c.chunk_id, "model": args.model, "temperature": args.temperature,
                   "latency_s": round(time.perf_counter() - t0, 2), "ttps": raw}
            cache[c.chunk_id] = rec
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        rec = cache[c.chunk_id]
        k, w = validate(c, rec["ttps"], s.technique_ids)
        kept += [b.model_dump() for b in k]
        why.update(w)
        raw_n += len(rec["ttps"])
        lat.append(rec.get("latency_s", 0.0))
        print(f"  [{i}/{len(sample)}] {c.chunk_id}: {len(rec['ttps'])} records, kept {len(k)}  ({rec.get('latency_s')}s)",
              flush=True)

    ids = {c.chunk_id for c in sample}
    ref = [json.loads(l) for l in (DERIVED / "report_ttps.jsonl").open()]
    ref = [r for r in ref if r["chunk_id"] in ids]
    out = {
        "model": args.model, "temperature": args.temperature, "chunks": len(sample),
        "records": raw_n, "kept": len(kept), "dropped": dict(why),
        "drop_rate": round(sum(why.values()) / raw_n, 3) if raw_n else None,
        "latency_s": {"mean": round(sum(lat) / len(lat), 2), "max": max(lat)},
        "reference": {"model": DEFAULT_MODEL, "kept": len(ref)},
        "agreement_exact": agreement(pairs(kept), pairs(ref)),
        "agreement_family": agreement(pairs(kept, True), pairs(ref, True)),
    }
    print(json.dumps(out, indent=1))
    (RUNS / "compare_extractors.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
