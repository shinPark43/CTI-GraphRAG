# CTI-GraphRAG: entity attribution with small open-source LLMs

CS 6338 Knowledge Engineering term project (Shinhyuk Park). A knowledge graph built
only from the [CTIConnect](https://github.com/peng-gao-lab/CTIConnect) `corpus_kb`
snapshot helps Llama-3-8B-Instruct map threat-report passages to MITRE ATT&CK
techniques (task ATA) and CWE weaknesses (task VCA). The design document is
`reports/ShinhyukPark_GraphRAG_DesignDoc.pdf`.

## Requirements

| Component | Version used | Notes |
| --- | --- | --- |
| Python | 3.13 | `requires-python >= 3.11`; gensim has no 3.14 wheels yet |
| Packages | `requirements.txt` (pinned, 450 lines), `requirements-dev.txt` (pytest) | exported from `uv.lock`; or use `uv sync` |
| Ollama | 0.35.0 | serves the answering model on `http://localhost:11434/v1` |
| Answering model | `llama3:8b-instruct-fp16` | Llama-3-8B-Instruct, F16, 8.0B parameters, 8,192-token context, ~16 GB; temperature 0 |
| Neo4j | 5.26 Community (Docker) | heap 2 GB |
| Neo4j plugins | Graph Data Science 2.13.14, APOC 5.26.32 | installed by `NEO4J_PLUGINS` in `docker-compose.yml`; n10s is not needed (property-graph design, no RDF import) |
| OpenAI API | `text-embedding-3-large` | all retrieval embeddings (cached in `.cache/` after the first run); `gpt-5.4-mini` only to rebuild the report layer or run RAGAS |

## Setup

```bash
# 1. Python environment (either one)
uv sync                                            # uv: exact versions from uv.lock
python3.13 -m venv .venv && . .venv/bin/activate \
  && pip install -r requirements.txt -r requirements-dev.txt && pip install -e . --no-deps

# 2. Configuration
cp .env.example .env        # set OPENAI_API_KEY and NEO4J_PASSWORD; LLM_* default to the local Ollama

# 3. Answering model (local)
ollama pull llama3:8b-instruct-fp16

# 4. Neo4j with GDS and APOC
docker compose up -d                               # container cti-neo4j, volume cti-neo4j-data

# 5. Data and checks
uv run python scripts/fetch_cticonnect.py          # pinned CTIConnect data, checksums verified
uv run python scripts/build_benchmark_files.py     # dev/test split, eval_queries.json, gold_triples.json
uv run pytest -q
```

With pip instead of uv, drop the `uv run` prefix from the commands below.

## Build the graph

```bash
uv run python scripts/build_report_layer.py --model gpt-5.4-mini-2026-03-17   # optional: data/derived is committed
uv run python scripts/compare_extractors.py --chunks 40  # same schema-constrained extractor on local Llama-3 vs the committed layer
uv run python scripts/load_graph.py --wipe          # applies schema.cypher, loads nodes, edges, vectors
uv run python scripts/check_graph_bridges.py        # Cypher candidates vs the in-memory reference
uv run python scripts/gds_analysis.py               # gds.cypher: degree, WCC, PageRank, Louvain, betweenness
uv run python scripts/extraction_audit.py score     # report-layer precision per confidence tier
```

## Run and evaluate

```bash
uv run python -m ctiattr.run --method sgs --subset dev --model llama3:8b-instruct-fp16 --name dev__sgs__llama3-8b-fp16
uv run python -m ctiattr.run --method vanilla_rag --subset dev --model llama3:8b-instruct-fp16 --name dev__vanilla_rag__llama3-8b-fp16
uv run python scripts/verify_triples.py dev__sgs__llama3-8b-fp16 dev__vanilla_rag__llama3-8b-fp16
uv run python scripts/ragas_eval.py                 # RAGAS on the 20 eval queries
uv run python scripts/graph_embed_recall.py         # graph-embedding candidate recall grid (no LLM)
uv run python scripts/gds_recall.py                 # PageRank / community as retrieval signals (no LLM)
uv run python scripts/cite_predictions.py dev__sgs__llama3-8b-fp16   # graph citations, checked in Neo4j
```

Runs are resumable and write `runs/<name>/` (predictions with full traces, scores).
The 306-question test split is reserved for the final evaluation.

## Chatbot and API

```bash
uv run streamlit run app/streamlit_app.py          # chatbot on http://localhost:8501
uv run uvicorn ctiattr.api:app --port 8000          # REST: POST /attribute, GET /graph/{id}, POST /verify, GET /examples
```

Paste a threat-report passage (or pick a dev-split example) and choose SGS, vanilla RAG or
closed-book. An SGS answer shows the chosen ATT&CK technique or CWE, the explanation, timings,
and five tabs: the **citations** (each graph path with its evidence, provenance and confidence,
the Cypher that returns it, and a button that checks it against the live graph), the cited
**subgraph**, the five **candidates** with their evidence cards and votes, **graph analytics**
for the answer node (PageRank and its percentile, Louvain community and its leading members),
and the **LLM calls**. Example questions come from the development split only.

## Repository checkpoint

| Component | Where |
| --- | --- |
| Environment: pinned packages, Neo4j + GDS/APOC, config template | `requirements.txt`, `requirements-dev.txt` (`uv export --no-hashes`), `docker-compose.yml`, `.env.example` |
| Graph schema: constraints, lookup and vector indexes | `schema.cypher` |
| Neo4j loader and Cypher candidate bridges | `scripts/load_graph.py`, `src/ctiattr/graph_db.py` |
| Snapshot-derived rules R1–R6 and candidate generators | `src/ctiattr/snapshot.py`, `src/ctiattr/candidates.py` |
| LLM extraction prompt, strict JSON schema and Pydantic validation (report layer) | `src/ctiattr/report_layer.py`, `src/ctiattr/schemas.py`, `data/derived/` |
| Edge provenance, extraction confidence tiers and their audit | `src/ctiattr/provenance.py`, `scripts/extraction_audit.py`, `data/derived/extraction_audit.csv` |
| Graph embeddings (propagated text, node2vec) | `src/ctiattr/graph_embed.py` |
| Graph Data Science: projection, PageRank, Louvain, WCC, betweenness, paths | `gds.cypher`, `src/ctiattr/gds.py`, `scripts/gds_analysis.py` |
| Attribution engine (SGS) and baselines (CB, VR, DtR) | `src/ctiattr/sgs.py`, `src/ctiattr/baselines.py` |
| Graph citations (traceable paths behind each answer) | `src/ctiattr/citations.py`, `scripts/cite_predictions.py` |
| Attribution service, FastAPI app, Streamlit chatbot | `src/ctiattr/service.py`, `src/ctiattr/api.py`, `app/streamlit_app.py` |
| Pydantic data models and API payloads | `src/ctiattr/schemas.py` |
| Official CTIConnect scorer (vendored, unmodified) | `src/ctiattr/vendor/cticonnect_metrics.py` |
| Gold triples (69) and eval queries (20) | `data/gold_triples.json`, `data/eval_queries.json` |
| Dev/test split by report | `data/splits/split_v1.json` |
| Triple verification, RAGAS, recall diagnostics | `scripts/verify_triples.py`, `scripts/ragas_eval.py`, `scripts/*_recall.py` |
| Parser and model tests (56) | `tests/` |
| Development-split results kept for the final report | `results/dev_results.md`, `reports/design_doc_rev81_with_results.md` |
| Run outputs behind those results (predictions with full traces, scores, citations, analyses) | `runs/` (see `runs/README.md`) |
| Identifier hallucination check | `scripts/hallucination_check.py` |
