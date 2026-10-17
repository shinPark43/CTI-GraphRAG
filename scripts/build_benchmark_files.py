#!/usr/bin/env python3
"""Build the Milestone 2 benchmark files from CTIConnect dev-split gold data.

    uv run python scripts/build_benchmark_files.py

- data/eval_queries.json : the 20 mini20 questions (10 ATA + 10 VCA)
- data/gold_triples.json : one (report, DESCRIBES_*, id) triple per dev question

Gold labels are CTIConnect's expert-verified answers (CC-BY-4.0, Cheng et al.,
KDD '26). Only the dev split is used, so the frozen test split stays unseen.
"""

from __future__ import annotations

import json
import re

from ctiattr.config import RAW, ROOT
from ctiattr.data import load_all, load_kb
from ctiattr.split import load_split

PREDICATE = {"ata": "DESCRIBES_TECHNIQUE", "vca": "DESCRIBES_WEAKNESS"}
TARGET_TYPE = {"ata": "AttackTechnique", "vca": "CWE"}
SOURCE = ("CTIConnect (Cheng et al., KDD '26), commit 554797d, CC-BY-4.0; "
          "https://github.com/peng-gao-lab/CTIConnect")

_QUOTE = re.compile(r"""security blog:\s*["'“‘](.+?)["'”’][.,;]?\s*(?:Please|Which|Could|Can|What|Identify|$)""",
                    re.S)


def passage(question: str) -> str:
    m = _QUOTE.search(question)
    return m.group(1).strip() if m else question


def main() -> None:
    split = load_split()
    qas = {qa.id: qa for qa in load_all()}
    names = {e.doc_id.upper(): e.title for kind in ("mitre", "cwe") for e in load_kb(kind)}
    reports = {}
    for line in open(RAW / "corpus_reports" / "preprocessed_reports.jsonl", encoding="utf-8"):
        r = json.loads(line)
        reports[f"BLOG-{r['id']}"] = {"title": r.get("title", ""), "url": r.get("link", ""),
                                      "published": r.get("publish_date", "")}

    def record(qa):
        gid = qa.gold[0]
        return {"report_id": qa.report_id, "report_title": reports.get(qa.report_id, {}).get("title", ""),
                "gold_id": gid, "gold_name": names.get(gid, "")}

    queries = []
    for qid in split["mini20"]:
        qa = qas[qid]
        queries.append({"id": qa.id, "task": qa.task, "question": qa.question,
                        "passage": passage(qa.question), **record(qa),
                        "target_type": TARGET_TYPE[qa.task]})

    triples, seen = [], set()
    for qid in split["dev"]:
        qa = qas[qid]
        key = (qa.report_id, PREDICATE[qa.task], qa.gold[0])
        ev = passage(qa.question)
        if key in seen:
            continue
        seen.add(key)
        rec = record(qa)
        triples.append({
            "id": f"gt-{len(triples) + 1:03d}",
            "subject": {"type": "ThreatReport", "id": qa.report_id, "name": rec["report_title"]},
            "predicate": PREDICATE[qa.task],
            "object": {"type": TARGET_TYPE[qa.task], "id": rec["gold_id"], "name": rec["gold_name"]},
            "evidence_text": ev,
            "source_question": qa.id,
        })

    meta = {"source": SOURCE, "split": "dev (data/splits/split_v1.json)"}
    (ROOT / "data" / "eval_queries.json").write_text(
        json.dumps({**meta, "subset": "mini20", "n": len(queries), "queries": queries},
                   indent=1, ensure_ascii=False) + "\n")
    (ROOT / "data" / "gold_triples.json").write_text(
        json.dumps({**meta, "n": len(triples), "triples": triples}, indent=1, ensure_ascii=False) + "\n")
    no_quote = sum(q["passage"] == q["question"] for q in queries)
    no_name = sum(not t["object"]["name"] for t in triples)
    print(f"eval_queries: {len(queries)} (passage not extracted: {no_quote})")
    print(f"gold_triples: {len(triples)} (missing gold name: {no_name})")


if __name__ == "__main__":
    main()
