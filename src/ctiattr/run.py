"""Run one method on one model over a set of questions, then score it.

    uv run python -m ctiattr.run --method vanilla_rag --model llama3:8b-instruct-fp16 \
        --subset all --limit 5 --name smoke

Writes runs/<name>/: config.json, predictions.jsonl (one record per question,
with the full trace) and scores.json. Re-running the same command resumes:
questions already in predictions.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from ctiattr.baselines import METHODS
from ctiattr.config import RUNS, EmbedConfig, LLMConfig
from ctiattr.data import TASK_KB, TASKS, load_all
from ctiattr.llm import ChatLLM
from ctiattr.retrieval import Embedder, Retriever
from ctiattr.score import score
from ctiattr.split import load_split


def select(subset: str, tasks: list[str], limit: int | None):
    qas = [qa for qa in load_all() if qa.task in tasks]
    if subset != "all":
        ids = set(load_split()[subset])
        qas = [qa for qa in qas if qa.id in ids]
    if limit:
        per_task: dict[str, int] = {}
        kept = []
        for qa in qas:
            if per_task.get(qa.task, 0) < limit:
                kept.append(qa)
                per_task[qa.task] = per_task.get(qa.task, 0) + 1
        qas = kept
    return qas


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--model", default=None, help="answering model (default: LLM_MODEL)")
    ap.add_argument("--subset", choices=["all", "dev", "test", "mini20"], default="dev")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="first N questions per task")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    llm_cfg = LLMConfig.from_env(args.model)
    name = args.name or f"{args.method}__{llm_cfg.model.replace(':', '-')}__{args.subset}"
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "model": llm_cfg.model, "llm_base_url": llm_cfg.base_url, "name": name}
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    qas = select(args.subset, args.tasks, args.limit)
    pred_path = out / "predictions.jsonl"
    done = {}
    if pred_path.exists():
        for line in pred_path.open():
            r = json.loads(line)
            done[r["id"]] = r
    todo = [qa for qa in qas if qa.id not in done]
    print(f"[{name}] {len(qas)} questions, {len(done)} already done, {len(todo)} to run")

    ctx = {"llm": ChatLLM(llm_cfg), "k": args.k}
    if args.method != "closed_book":
        ctx["retriever"] = Retriever(Embedder(EmbedConfig.from_env()))
        for t in args.tasks:  # build indexes up front, not inside worker threads
            ctx["retriever"].index(TASK_KB[t])

    method = METHODS[args.method]
    lock = threading.Lock()

    def one(qa):
        t0 = time.perf_counter()
        try:
            text, trace = method(qa, ctx)
            err = None
        except Exception:  # noqa: BLE001 - record and score as empty
            text, trace, err = "", {}, traceback.format_exc(limit=3)
        return {"id": qa.id, "task": qa.task, "gold": qa.gold, "prediction": text,
                "error": err, "seconds": round(time.perf_counter() - t0, 2), **trace}

    with pred_path.open("a") as f, ThreadPoolExecutor(args.concurrency) as ex:
        futures = [ex.submit(one, qa) for qa in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[rec["id"]] = rec
            if rec["error"]:
                print(f"  ERROR {rec['id']}: {rec['error'].strip().splitlines()[-1]}")
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)

    result = score({i: r["prediction"] for i, r in done.items()}, qas)
    result["n_errors"] = sum(1 for qa in qas if done.get(qa.id, {}).get("error"))
    (out / "scores.json").write_text(json.dumps(result, indent=1) + "\n")
    o = result["overall"]
    print(f"[{name}] F1={o['f1']:.3f} P={o['precision']:.3f} R={o['recall']:.3f} "
          f"EM={o['exact_match']:.3f} mean_ids={result['mean_pred_ids']} errors={result['n_errors']}")
    for t, s in result["per_task"].items():
        print(f"  {t}: n={s['n']} F1={s['f1']:.3f} P={s['precision']:.3f} R={s['recall']:.3f}")


if __name__ == "__main__":
    main()
