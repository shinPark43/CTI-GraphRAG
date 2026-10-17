#!/usr/bin/env python3
"""RAGAS evaluation of SGS (with its explanation step) and vanilla RAG on the 20 eval queries.

For each query:
  user_input          the CTIConnect question
  retrieved_contexts  what the answering model was shown (SGS: the 5 evidence cards;
                      VR: the 5 KB records, 600 characters each, as CTIConnect formats them)
  response            the final answer text (SGS: ID + title + explanation)
  reference           CTIConnect's reference answer for the question
Metrics (RAGAS 0.4 collections API): Faithfulness, AnswerRelevancy, ContextRecall.
Judge: gpt-5.4-mini; embeddings: text-embedding-3-large (both OpenAI).

SGS choices are reused from runs/dev__sgs__llama3-8b-fp16 (same frozen run); only
the explanations are generated here, and the official F1 is checked to be unchanged.

    uv run python scripts/ragas_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json

from openai import AsyncOpenAI, OpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

from ctiattr.config import RAW, RUNS, EmbedConfig, LLMConfig
from ctiattr.data import KBEntry, TASK_KB, load_all, load_kb
from ctiattr.llm import ChatLLM
from ctiattr.retrieval import Embedder, Hit, format_candidates
from ctiattr.sgs import SGS
from ctiattr.split import load_split
from ctiattr.vendor.cticonnect_metrics import score_id_item

METRICS = ("faithfulness", "answer_relevancy", "context_recall")
USAGE = {"calls": 0, "input": 0, "output": 0}
PRICE_PER_M = {"input": 0.75, "output": 4.50}  # gpt-5.4-mini list price, USD per million tokens


def references() -> dict[str, str]:
    out = {}
    for task in ("ata", "vca"):
        for line in open(RAW / "data" / "entity_attribution" / f"{task}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            out[d["id"]] = d["answer"]
    return out


def reasoning_client(key: str) -> AsyncOpenAI:
    """AsyncOpenAI client adapted for GPT-5-family judges.

    RAGAS 0.4.3 maps max_tokens -> max_completion_tokens only when it recognizes a
    reasoning model, and its check parses "gpt-5.4-mini" as version "5.4", which
    fails. This adapter applies the same mapping (and drops sampling parameters
    these models reject) for any model name.
    """
    client = AsyncOpenAI(api_key=key)
    create = client.chat.completions.create

    async def patched(*args, **kw):
        if "max_tokens" in kw:
            kw["max_completion_tokens"] = kw.pop("max_tokens")
        for p in ("temperature", "top_p"):
            kw.pop(p, None)
        resp = await create(*args, **kw)
        if getattr(resp, "usage", None):  # judge token accounting for the cost report
            USAGE["calls"] += 1
            USAGE["input"] += resp.usage.prompt_tokens or 0
            USAGE["output"] += resp.usage.completion_tokens or 0
        return resp

    client.chat.completions.create = patched
    return client


async def score_all(samples: list[dict], judge: str, emb_model: str, key: str, concurrency: int = 4) -> list[dict]:
    llm = llm_factory(judge, client=reasoning_client(key))
    emb = embedding_factory("openai", model=emb_model, client=OpenAI(api_key=key))
    scorers = {"faithfulness": Faithfulness(llm=llm),
               "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
               "context_recall": ContextRecall(llm=llm)}
    sem = asyncio.Semaphore(concurrency)

    async def one(s):
        async with sem:
            out = {}
            for name, scorer in scorers.items():
                kw = {"user_input": s["user_input"]}
                if name != "context_recall":
                    kw["response"] = s["response"]
                if name != "answer_relevancy":
                    kw["retrieved_contexts"] = s["retrieved_contexts"]
                if name == "context_recall":
                    kw["reference"] = s["reference"]
                try:
                    out[name] = float((await scorer.ascore(**kw)).value)
                except Exception as e:  # noqa: BLE001 - record, do not drop the sample
                    out[name] = None
                    out[f"{name}_error"] = f"{type(e).__name__}: {e}"[:200]
            return {**s, **out}

    return await asyncio.gather(*(one(s) for s in samples))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b-instruct-fp16")
    ap.add_argument("--judge", default="gpt-5.4-mini-2026-03-17")
    ap.add_argument("--sgs-run", default="dev__sgs__llama3-8b-fp16")
    ap.add_argument("--vr-run", default="dev__vanilla_rag__llama3-8b-fp16")
    args = ap.parse_args()

    ids = load_split()["mini20"]
    qa = {q.id: q for q in load_all()}
    ref = references()
    sgs_runs = {json.loads(l)["id"]: json.loads(l) for l in (RUNS / args.sgs_run / "predictions.jsonl").open()}
    vr_runs = {json.loads(l)["id"]: json.loads(l) for l in (RUNS / args.vr_run / "predictions.jsonl").open()}
    kb = {kind: {e.doc_id.upper(): e for e in load_kb(kind)} for kind in ("mitre", "cwe")}

    emb_cfg = EmbedConfig.from_env()
    sgs = SGS(Embedder(emb_cfg))
    llm = ChatLLM(LLMConfig.from_env(args.model))

    samples = []
    for i in ids:
        q, r = qa[i], sgs_runs[i]
        explanation, _ = sgs.explain(llm, q, r["chosen"], r["cards"][r["chosen"]])
        response = r["prediction"] + "\n\n" + explanation
        before = score_id_item(r["prediction"], q.ground_truth).f1
        after = score_id_item(response, q.ground_truth).f1
        assert before == after, f"{i}: explanation changed F1 {before} -> {after}"
        samples.append({"system": "SGS", "id": i, "task": q.task, "f1": after, "user_input": q.question,
                        "retrieved_contexts": [r["cards"][c] for c in r["candidates"]],
                        "response": response, "reference": ref[i]})

        v = vr_runs[i]
        hits = [Hit(d, 0.0, kb[TASK_KB[q.task]][d].title, kb[TASK_KB[q.task]][d].text)
                for d in v["retrieved"][0] if d in kb[TASK_KB[q.task]]]
        samples.append({"system": "VR", "id": i, "task": q.task,
                        "f1": score_id_item(v["prediction"], q.ground_truth).f1, "user_input": q.question,
                        "retrieved_contexts": format_candidates(hits).split("\n\n"),
                        "response": v["prediction"], "reference": ref[i]})
    print(f"explanations generated for {len(ids)} queries; official F1 unchanged by the explanation step")

    scored = asyncio.run(score_all(samples, args.judge, emb_cfg.model, emb_cfg.api_key))
    out = RUNS / "ragas__mini20.jsonl"
    out.write_text("".join(json.dumps(s) + "\n" for s in scored))

    print(f"\nRAGAS on the 20 eval queries (judge {args.judge})")
    print(f"{'system':8}{'task':6}{'n':>3} {'F1':>6} " + " ".join(f"{m:>17}" for m in METRICS))
    for system in ("VR", "SGS"):
        for task in ("ata", "vca", "all"):
            rows = [s for s in scored if s["system"] == system and (task == "all" or s["task"] == task)]
            mean = lambda k: sum(s[k] for s in rows if s[k] is not None) / max(1, sum(s[k] is not None for s in rows))
            print(f"{system:8}{task:6}{len(rows):3} {mean('f1'):6.2f} " + " ".join(f"{mean(m):17.2f}" for m in METRICS))
    errors = sum(1 for s in scored for m in METRICS if s[m] is None)
    print(f"metric errors: {errors}; per-sample scores -> {out}")
    cost = sum(USAGE[k] * PRICE_PER_M[k] / 1e6 for k in PRICE_PER_M)
    print(f"judge usage: {USAGE['calls']} calls, {USAGE['input']:,} input + {USAGE['output']:,} output tokens "
          f"(about ${cost:.2f} at list price)")


if __name__ == "__main__":
    main()
