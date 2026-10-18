#!/usr/bin/env python3
"""Build the report layer (CTIFoundry Steps 0 and 2) and measure its coverage.

    uv run python scripts/build_report_layer.py --model gpt-5.4-mini-2026-03-17
    uv run python scripts/build_report_layer.py --backend ollama --model llama3:8b-instruct-fp16 --temperature 0

Writes data/derived/report_chunks.jsonl, data/derived/report_ttps.jsonl and a
build log. Extraction calls are cached in .cache/report_ttps_raw.jsonl, so a
re-run costs nothing. Coverage is measured on the dev split only, excluding each
question's own source report.
"""

from __future__ import annotations

import argparse
import datetime
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from ctiattr.config import CACHE, ROOT, EmbedConfig, LLMConfig
from ctiattr.data import load_all
from ctiattr.report_layer import chunk_report, extract, load_reports, to_dict, validate
from ctiattr.snapshot import Snapshot
from ctiattr.split import load_split

OUT = ROOT / "data" / "derived"
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"


def extraction_cache(model: str):
    """Raw replies per chunk; the committed build's cache keeps its original name."""
    name = "report_ttps_raw" if model == DEFAULT_MODEL else f"report_ttps_raw__{model.replace(':', '-')}"
    return CACHE / f"{name}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", choices=["openai", "ollama"], default="openai",
                    help="ollama = the local OpenAI-compatible endpoint (LLM_BASE_URL)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="omitted = provider default (the committed gpt-5.4-mini build used the default)")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    s = Snapshot()
    tactics = sorted({t for tid in s.technique_ids for t in s.tactics(tid)})
    reports = load_reports()
    chunks = [c for rid, text in sorted(reports.items(), key=lambda kv: int(kv[0].split("-")[1]))
              for c in chunk_report(rid, text)]
    lens = sorted(len(c.text) for c in chunks)
    print(f"Step 0: {len(reports)} reports -> {len(chunks)} chunks "
          f"(chars min {lens[0]}, median {lens[len(lens) // 2]}, max {lens[-1]})")

    CACHE.mkdir(exist_ok=True)
    cache_path = extraction_cache(args.model)
    # other models write beside the committed layer, never over it
    out = OUT if args.model == DEFAULT_MODEL else OUT / args.model.replace(":", "-")
    cache = {}
    if cache_path.exists():
        cache = {json.loads(l)["chunk_id"]: json.loads(l)["ttps"] for l in cache_path.open()}
    client = (OpenAI(base_url=LLMConfig.from_env().base_url, api_key=LLMConfig.from_env().api_key, timeout=900)
              if args.backend == "ollama" else OpenAI(api_key=EmbedConfig.from_env().api_key))
    lock = threading.Lock()

    fresh = []

    def run(c):
        if c.chunk_id not in cache:
            fresh.append(c.chunk_id)
            raw = extract(client, args.model, c, tactics, args.temperature)
            with lock:
                cache[c.chunk_id] = raw
                with cache_path.open("a") as f:
                    f.write(json.dumps({"chunk_id": c.chunk_id, "model": args.model, "ttps": raw}) + "\n")
        return c, cache[c.chunk_id]

    with ThreadPoolExecutor(args.concurrency) as ex:
        results = list(ex.map(run, chunks))

    kept, drops, raw_n = [], Counter(), 0
    for c, raw in results:
        k, why = validate(c, raw, s.technique_ids)
        kept += k
        drops.update(why)
        raw_n += len(raw)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report_chunks.jsonl").write_text("".join(json.dumps(to_dict(c)) + "\n" for c in chunks))
    (out / "report_ttps.jsonl").write_text("".join(json.dumps(to_dict(b)) + "\n" for b in kept))
    print(f"Step 2: {raw_n} extracted, {len(kept)} kept; dropped {dict(drops)}")
    print(f"  distinct techniques with >=1 example: {len({b.technique_id for b in kept})}; "
          f"chunks with >=1 behavior: {len({b.chunk_id for b in kept})}/{len(chunks)}")

    # coverage on dev ATA, never counting the question's own source report
    dev = set(load_split()["dev"])
    ata = [q for q in load_all() if q.task == "ata" and q.id in dev]
    exact = fam = 0
    for q in ata:
        others = [b for b in kept if b.report_id != q.report_id]
        g = q.gold[0]
        exact += any(b.technique_id == g for b in others)
        fam += any(b.technique_id.split(".")[0] == g.split(".")[0] for b in others)
    print(f"Coverage (dev ATA, n={len(ata)}, own report excluded): gold technique has >=1 example "
          f"{exact}/{len(ata)}; same technique family {fam}/{len(ata)}")
    # built_on = date of the extraction calls; a rebuild entirely from the cache keeps the old date
    prev = out / "report_layer_build.json"
    built_on = (json.loads(prev.read_text()).get("built_on") if prev.exists() and not fresh else None) \
        or datetime.date.today().isoformat()
    log = {"model": args.model, "built_on": built_on, "chunks": len(chunks), "extracted": raw_n, "kept": len(kept),
           "dropped": dict(drops), "dev_ata_exact_coverage": f"{exact}/{len(ata)}",
           "dev_ata_family_coverage": f"{fam}/{len(ata)}"}
    (out / "report_layer_build.json").write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
