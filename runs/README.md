# Development-split results

Outputs of the experiments summarized in `results/dev_results.md`. Everything here is on
the 73-question development split (30 ATA, 43 VCA), which was also used to choose
configurations; the 306-question test split has not been run. Other runs (smoke tests,
the `mini20` pilot, diagnostics) stay local.

| Path | What | Made by |
| --- | --- | --- |
| `dev__closed_book__llama3-8b-fp16/` | Closed-book baseline (CB) | `python -m ctiattr.run --method closed_book` |
| `dev__vanilla_rag__llama3-8b-fp16/` | Vanilla RAG baseline (VR) | `python -m ctiattr.run --method vanilla_rag` |
| `dev__sgs__llama3-8b-fp16/` | SGS, in-memory retrieval; `citations.jsonl` + `citations_summary.json` | `python -m ctiattr.run --method sgs`, `scripts/cite_predictions.py` |
| `dev__sgs-neo4j__llama3-8b-fp16/` | SGS with candidates from Cypher queries on Neo4j | `python -m ctiattr.run --method sgs --backend neo4j` |
| `hallucination__dev.json` | Fabricated / missing / hedged identifiers per method | `scripts/hallucination_check.py` |
| `ragas__mini20.jsonl` | RAGAS faithfulness, answer relevancy, context recall (20 eval queries) | `scripts/ragas_eval.py` |
| `graph_embed_recall__dev.json`, `gds_recall__dev.json` | Candidate recall@k for graph embeddings and GDS signals | `scripts/graph_embed_recall.py`, `scripts/gds_recall.py` |
| `gds_analysis.json` | Topology, PageRank, Louvain, WCC, betweenness, example path | `scripts/gds_analysis.py` |
| `extraction_audit.json`, `compare_extractors.json` | Report-layer precision by confidence tier; local Llama-3 extraction vs gpt-5.4-mini | `scripts/extraction_audit.py`, `scripts/compare_extractors.py` |

Each run directory holds `config.json` (settings), `predictions.jsonl` (one record per
question with the full trace: candidates, evidence cards, prompts, outputs, latency),
`scores.json` (official CTIConnect scorer) and `triple_scores.json`.

Questions and passages in these files come from the CTIConnect benchmark
(https://github.com/peng-gao-lab/CTIConnect), CC-BY-4.0, © the CTIConnect authors.
