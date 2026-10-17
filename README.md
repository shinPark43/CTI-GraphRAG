# CTI-GraphRAG: entity attribution with small open-source LLMs

CS 6338 Knowledge Engineering term project (Shinhyuk Park). A knowledge graph built
only from the [CTIConnect](https://github.com/peng-gao-lab/CTIConnect) `corpus_kb`
snapshot helps Llama-3-8B-Instruct map threat-report passages to MITRE ATT&CK
techniques (task ATA) and CWE weaknesses (task VCA). The design document is
`reports/ShinhyukPark_GraphRAG_DesignDoc.pdf`.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/), Ollama with
`llama3:8b-instruct-fp16`, and Docker for Neo4j.

```bash
uv sync
cp .env.example .env        # fill in OPENAI_API_KEY (embeddings) and NEO4J_PASSWORD
uv run python scripts/fetch_cticonnect.py          # pinned CTIConnect data, checksums verified
uv run python scripts/build_benchmark_files.py     # dev/test split, eval_queries.json, gold_triples.json
uv run pytest -q
```

## Build the graph

```bash
docker run -d --name cti-neo4j -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
    -e NEO4J_AUTH=neo4j/<password> -v cti-neo4j-data:/data neo4j:5.26
uv run python scripts/build_report_layer.py --model gpt-5.4-mini-2026-03-17   # optional: data/derived is committed
uv run python scripts/load_graph.py --wipe          # applies schema.cypher, loads nodes, edges, vectors
uv run python scripts/check_graph_bridges.py        # Cypher candidates vs the in-memory reference
```

## Run and evaluate

```bash
uv run python -m ctiattr.run --method sgs --subset dev --model llama3:8b-instruct-fp16 --name dev__sgs__llama3-8b-fp16
uv run python -m ctiattr.run --method vanilla_rag --subset dev --model llama3:8b-instruct-fp16 --name dev__vanilla_rag__llama3-8b-fp16
uv run python scripts/verify_triples.py dev__sgs__llama3-8b-fp16 dev__vanilla_rag__llama3-8b-fp16
uv run python scripts/ragas_eval.py                 # RAGAS on the 20 eval queries
uv run python scripts/graph_embed_recall.py         # graph-embedding candidate recall grid (no LLM)
```

Runs are resumable and write `runs/<name>/` (predictions with full traces, scores).
The 306-question test split is reserved for the final evaluation.

## Repository checkpoint

| Component | Where |
| --- | --- |
| Graph schema: constraints, lookup and vector indexes | `schema.cypher` |
| Neo4j loader and Cypher candidate bridges | `scripts/load_graph.py`, `src/ctiattr/graph_db.py` |
| Snapshot-derived rules R1–R6 and candidate generators | `src/ctiattr/snapshot.py`, `src/ctiattr/candidates.py` |
| LLM extraction prompt (report layer) and validator | `src/ctiattr/report_layer.py`, `data/derived/` |
| Graph embeddings (propagated text, node2vec) | `src/ctiattr/graph_embed.py` |
| Attribution engine (SGS) and baselines (CB, VR, DtR) | `src/ctiattr/sgs.py`, `src/ctiattr/baselines.py` |
| Pydantic data models and API payloads | `src/ctiattr/schemas.py` |
| Official CTIConnect scorer (vendored, unmodified) | `src/ctiattr/vendor/cticonnect_metrics.py` |
| Gold triples (69) and eval queries (20) | `data/gold_triples.json`, `data/eval_queries.json` |
| Dev/test split by report | `data/splits/split_v1.json` |
| Triple verification, RAGAS, recall diagnostics | `scripts/verify_triples.py`, `scripts/ragas_eval.py`, `scripts/*_recall.py` |
| Parser and model tests (39) | `tests/` |
| Development-split results kept for the final report | `results/dev_results.md`, `reports/design_doc_rev81_with_results.md` |

Not built yet: the FastAPI service and Streamlit chatbot specified in Section 8 of the design document.
