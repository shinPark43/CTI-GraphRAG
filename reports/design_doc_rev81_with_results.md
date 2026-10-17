<!-- Snapshot of the Claude Docs design document at rev 81 (2026-10-17), before results were moved out of the Milestone 2 version. Embedded charts are not exported; their data are in runs/ (snapshot_recall__dev.jsonl, mini20__*). Source: https://claude.ai/code/artifact/4f01d902-42f3-42d9-90c5-07bcde6a11d3 -->
# CTI Entity Attribution with Small Open-Source LLMs

**System Architecture & Design Document — CS 6338 Knowledge Engineering, Milestone 2**

Oct 16, 2026 · @Shin

## 1. Executive Summary & System Requirements

This project builds a knowledge-graph-guided attribution engine that helps a small open-source LLM (Llama-3-8B-Instruct) map threat-report passages to MITRE ATT&CK techniques (CTIConnect task ATA) and CWE weaknesses (task VCA). On CTIConnect, LLaMA-3-8B reaches only 0.28 F1 on ATA and 0.40 on VCA, against 0.90 and 0.76 for GPT-5; the benchmark authors identify attribution as the one task family bottlenecked by model capability rather than retrieval.

**Scope change (approved by Dr. Dogdu, Oct 16, 2026).** The proposal's four-task, multi-hop GraphRAG evaluation is narrowed to entity attribution only. Entity-linking tasks (RCM, ESD, ATD) already reach about 0.97 F1 with domain-specific retrieval and are single-hop, so they cannot show what a graph adds. Multi-hop questions are no longer required.

**Research question.** Can a cybersecurity knowledge graph, used as a reasoning scaffold, close the gap between small open-source models and frontier models on CTI entity attribution?

**Domain and persona.** A SOC or threat-intelligence analyst reads a vendor report and needs the ATT&CK technique or CWE weakness a passage describes, with the reasoning and the KB entry that support it. The system must run on local hardware (a DGX Spark and a MacBook) without sending reports to a commercial LLM.

### Functional requirements

| ID | Requirement |
| --- | --- |
| FR1 | Given a report passage, return exactly one ATT&CK technique ID (ATA) or CWE ID (VCA), matching the benchmark's single-gold format |
| FR2 | Ground every answer in a KB entry: the returned ID must exist in the knowledge graph and its name and description are shown with it |
| FR3 | Show the candidate set and the graph path used (tactic → technique → sub-technique, or passage → CAPEC/CVE → CWE) as the answer's citation |
| FR4 | Show each candidate's tactics, platforms and examples on its evidence card; never filter the answer space by domain or retired status |
| FR5 | Use only CTIConnect's corpus\_kb snapshot as evidence: no outside data (MITRE releases, NVD) and none of the benchmark's construction files, so results stay comparable with the paper |
| FR6 | Log every LLM call (prompt, output, tokens, latency) and every retrieved candidate, for error analysis |

### Non-functional requirements

| ID | Requirement | Target |
| --- | --- | --- |
| NFR1 | Local inference only for the answering model | Ollama on DGX Spark or Mac; no report text sent to a hosted LLM |
| NFR2 | Answer latency per question | under 10 s for a single-call method on the DGX Spark; up to 30 s for multi-step methods |
| NFR3 | Reproducibility | temperature 0, pinned dataset commit, model digest and quantization recorded per run |
| NFR4 | Cost | embeddings cached on disk; one-time KB embedding about $0.09 |
| NFR5 | Evaluation integrity | frozen test split, never used for prompt or method development |

## 2. End-to-End System Architecture

The system has an offline build stage that turns official records into a validated Neo4j graph, and a per-question stage in which the graph produces candidates and the small LLM only chooses among them.

&#91;embedded content: system architecture · build time and query time\]

Every ID the system can answer was written by the loader from an official record; the LLM's output is restricted to IDs the graph handed it.

### Data flow

| From → To | Input | Output | Protocol |
| --- | --- | --- | --- |
| Sources → loader | CTIConnect `corpus_kb` JSONL (pinned commit) | validated node and edge records | file read, sha256 check |
| Loader → Neo4j | node and edge records, rule results R1, R2, R4 | graph with constraints and indexes | Bolt (neo4j Python driver), batched `UNWIND` writes |
| Embedder → Neo4j | text(v) per answerable node | 3,072-d vectors h(v) | OpenAI embeddings API, disk cache; Bolt write |
| Chatbot → FastAPI | passage, taxonomy, optional question | `AttributionRequest` JSON | HTTP POST `/attribute` |
| Parse → Retrieve | quoted passage, regex-matched IDs | query text, validated IDs | in-process |
| Retrieve ↔ Neo4j | query vector, query text | top-20 candidates with RRF scores | Cypher: vector indexes over techniques, CWEs, CAPECs and CVEs |
| Expand ↔ Neo4j | top-5 candidate IDs, cited CVEs | evidence cards with CAPEC, CVE and report examples (R4–R6) | Cypher, depth ≤ 3 |
| Select → Ollama | candidate list with definitions | one ID (VCA: majority over 5 rotations) | OpenAI-compatible chat API, JSON-schema enum, temperature 0 |
| Answer → chatbot / runner | chosen ID, graph path, LLM calls | `AttributionResponse` JSON | HTTP response; runner writes `predictions.jsonl` |
| Runner → scorer | prediction text per question | P / R / F1, exact match | CTIConnect `metrics.py`, in-process |

## 3. Knowledge-Graph Schema & Construction

The graph is built only from CTIConnect's `corpus_kb` snapshot (commit 554797d, dated 2025-09-01), with no outside data, so every result is directly comparable with the paper. The structure the baselines ignore is parsed out of the four snapshot files deterministically, with no LLM in the build; its edge counts match CTIFoundry's for the same snapshot (518 sub-technique links; 1,212 distinct CAPEC → CWE and 271 CAPEC → ATT&CK links, which CTIFoundry counts as 1,214 and 272 because three CAPEC records list a link twice).

### What each snapshot file contributes

| File | Records | What the graph uses |
| --- | --- | --- |
| `mitre.jsonl` | 1,076 techniques | title, tactics (`kill_chain_phases`), platforms, detection advice; domain and retired status are derived (Section 4) |
| `cwe.jsonl` | 1,342 weaknesses | title and `contents` (consequences, modes of introduction, mitigations, detection); 412 have empty contents |
| `capec.jsonl` | 615 attack patterns | a full `Description` in attack language on every record; 450 link to CWEs, 177 link to ATT&CK techniques |
| `cve.jsonl` | 3,011 CVEs | 2,751 have a description and a CWE in the snapshot: labelled examples covering 260 CWEs |
| `corpus_reports` | 321 reports | title, vendor and date for `ThreatReport` nodes |

`contents` serves as each ATT&CK and CWE record's description in CTIConnect, but it holds detection, mitigation and consequence text rather than a definition. CAPEC descriptions and CVE descriptions are the snapshot's natural-language bridge: they describe attacks and vulnerabilities in words much closer to report prose.

### Excluded, and why

- **Outside data** (MITRE ATT&CK STIX, CWE XML, NVD): adds evidence the benchmark's systems did not have. A cited CVE's NVD entry, for example, names the answer directly.
- **The benchmark's `construction/` folder**: its seeds, including `cwe_xrefs.jsonl`, were used to generate CTIConnect's questions, and its `b2f_*` files hold the answer annotations. As a consequence the graph has no CWE parent–child hierarchy.

### Node types

| Label | Key | Attributes |
| --- | --- | --- |
| `AttackTechnique` | `id` (T1059, T1059.001) | title, tactics, platforms, detection, domain, replaced\_by, answerable |
| `Tactic` | `name` (defense-evasion) | — |
| `CWE` | `id` (CWE-79) | title, contents |
| `CAPEC` | `id` (CAPEC-66) | name, description |
| `CVE` | `id` (CVE-2012-4687) | description |
| `ThreatReport` | `id` (BLOG-10) | title, url, published, vendor |
| `Passage` | `id` (question id) | text, report\_id |

### Relationship predicates

| Predicate | From → To | Derived from | Edges |
| --- | --- | --- | --- |
| `SUBTECHNIQUE_OF` | AttackTechnique → AttackTechnique | ID pattern (T1059.001 → T1059) | 518 |
| `IN_TACTIC` | AttackTechnique → Tactic | `kill_chain_phases` | 1,374 |
| `REPLACED_BY` | AttackTechnique → AttackTechnique | retired-ID rule R2 (Section 4) | 103 |
| `EXPLOITS_WEAKNESS` | CAPEC → CWE | CAPEC `Related_Weaknesses` | 1,212 |
| `MAPS_TO_TECHNIQUE` | CAPEC → AttackTechnique | CAPEC `Taxonomy_Mappings` (ATTACK entries) | 271 |
| `HAS_WEAKNESS` | CVE → CWE | CVE `weaknesses` | 3,286 |
| `CONTAINS` | ThreatReport → Passage | question source | 379 |
| `DESCRIBES_TECHNIQUE`, `DESCRIBES_WEAKNESS` | Passage → AttackTechnique / CWE | CTIConnect gold, **evaluation only** | 379 |

The gold `DESCRIBES_*` edges live in a separate evaluation database and are never loaded into the graph the engine queries.

### Query-time extraction

The only extraction from text happens on the question:

- **The quoted passage** is extracted by regex (all 20 eval queries and 68 of 69 gold-triple passages; one question adds evidence after the quote, so the whole question is kept).
- **Identifiers** (CVE, CWE, ATT&CK IDs) are captured by regex, never by the LLM, and validated against the graph.
- **No chunking**: passages are short (median 358 characters) and processed whole.

### Entity resolution: names to IDs

Small models recall titles more reliably than numbers. In the preliminary run, Llama-3-8B paired "Exploit Public-Facing Application" with T1059.001 (it is T1190). So a generated ID is never trusted on its own:

1. **Exact match**: the ID exists in the graph and its title matches the generated name after normalization.
2. **Title lookup**: normalized title equality within the target taxonomy.
3. **Retired IDs**: follow `REPLACED_BY` to the current sub-technique.
4. **Embedding fallback**: nearest title by cosine similarity, accepted only above a threshold τ tuned on the dev split (initial τ = 0.80); below τ the answer is marked unresolved.

### Relation to CTIFoundry's build phase

The build follows CTIFoundry's rule that the mechanism follows the evidence (Cheng et al., 2026, §3.2, App. A): structure the snapshot already records is parsed with no model, prose is handled by an LLM bracketed by deterministic checks, and paraphrase matching is left to an embedding model.

| CTIFoundry build artifact | This project |
| --- | --- |
| Ontology graph parsed from the four knowledge bases | reproduced; edge counts match (518 sub-technique links; 1,212 / 271 distinct CAPEC links vs CTIFoundry's 1,214 / 272 with duplicates) |
| Step 0: clause-bounded chunks of 200–900 characters with offsets | 650 chunks from the 321 report summaries (CTIFoundry: 688); every offset re-verifies against the source |
| Step 1: typed entity mentions (8 STIX types) | not built: actor and malware names serve multi-document synthesis, which is out of scope |
| Step 2: TTP grounding with prompt G.3 | built with that prompt (tactic list from the snapshot) and gpt-5.4-mini-2026-03-17 |
| Step 3: entity resolution (union-find, judge G.4) | not built, same reason as Step 1 |
| Validator: no invented IDs, no unverifiable spans | applied: an ID must exist in the snapshot and the behavior phrase must occur verbatim in its chunk |

**Report layer in numbers.** gpt-5.4-mini extracted 3,741 attacker behaviors; 3,512 pass the checks (6 unknown IDs and 223 non-verbatim phrases dropped), covering 353 techniques and 627 of 650 chunks. The question passages never appear in their own report's summary (0 of 73 dev passages), so the layer cannot answer a question from its source text; it serves as labelled examples of report wording for each technique, always excluding the question's own report.

**Measured use (dev ATA, n = 30, all 1,076 techniques).** 29 of 30 gold techniques have at least one example from another report. Fusing a vote over the 30 nearest example phrases with retrieval raises recall@3 from 0.43 to 0.60 and recall@5 from 0.57 to 0.63, with no domain filter.

## 4. Graph Storage, Enhancement & Reasoning

The graph lives in Neo4j 5, which holds the typed nodes and edges and vector indexes over every text the engine searches. The reasoning layer adds three bridges from report wording to taxonomy entries (CAPEC patterns, CVE examples and report-layer behaviors) and derives domain, retired status and tactics as attributes. It never filters candidates by those attributes: on the dev split filtering raised recall, but only by assuming what the answers look like.

### Database initialization (`schema.cypher`)

```cypher
// uniqueness: one node per snapshot identifier
CREATE CONSTRAINT technique_id IF NOT EXISTS FOR (t:AttackTechnique) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT tactic_name IF NOT EXISTS FOR (t:Tactic) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT cwe_id IF NOT EXISTS FOR (w:CWE) REQUIRE w.id IS UNIQUE;
CREATE CONSTRAINT capec_id IF NOT EXISTS FOR (c:CAPEC) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT cve_id IF NOT EXISTS FOR (v:CVE) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT report_id IF NOT EXISTS FOR (r:ThreatReport) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (k:ReportChunk) REQUIRE k.id IS UNIQUE;
CREATE CONSTRAINT behavior_id IF NOT EXISTS FOR (b:ReportBehavior) REQUIRE b.id IS UNIQUE;

// lookup indexes: title resolution and card attributes
CREATE INDEX technique_title IF NOT EXISTS FOR (t:AttackTechnique) ON (t.title_norm);
CREATE INDEX cwe_title IF NOT EXISTS FOR (w:CWE) ON (w.title_norm);
CREATE INDEX technique_domain IF NOT EXISTS FOR (t:AttackTechnique) ON (t.domain);
CREATE INDEX chunk_report IF NOT EXISTS FOR (k:ReportChunk) ON (k.report_id);

// vector indexes: text-embedding-3-large, 3,072 dimensions, cosine
CREATE VECTOR INDEX technique_emb IF NOT EXISTS FOR (t:AttackTechnique) ON (t.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX cwe_emb IF NOT EXISTS FOR (w:CWE) ON (w.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX capec_emb IF NOT EXISTS FOR (c:CAPEC) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX cve_emb IF NOT EXISTS FOR (v:CVE) ON (v.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX behavior_emb IF NOT EXISTS FOR (b:ReportBehavior) ON (b.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};

// example: the CAPEC bridge for one passage vector, exact similarity
// MATCH (c:CAPEC) WITH c, vector.similarity.cosine(c.embedding, $v) AS score
// ORDER BY score DESC LIMIT 10
// MATCH (c)-[:EXPLOITS_WEAKNESS]->(w:CWE)
// RETURN w.id AS cwe, sum(2 * score - 1) AS bridge_score ORDER BY bridge_score DESC;
```

The report layer is stored as `(ThreatReport)-[:HAS_CHUNK]->(ReportChunk)-[:HAS_BEHAVIOR]->(ReportBehavior)-[:GROUNDS]->(AttackTechnique)`. `scripts/load_graph.py` applies `schema.cypher` (17 statements) and loads 10,286 nodes, 9,296 of them with embeddings, from the on-disk embedding cache in about 30 seconds, checking every edge count against the Python derivation. Neo4j 5.26 runs locally in Docker.

SGS can take its candidates from Neo4j (`--backend neo4j`): each bridge is one Cypher query using exact `vector.similarity.cosine`. On the dev split the Cypher candidates match the in-memory implementation on 65 of 73 top-5 lists, 70 of 73 sets and 72 of 73 gold-in-top-5 outcomes; the remaining differences are near-ties between 32-bit (numpy) and 64-bit (Neo4j) similarity. The HNSW vector indexes serve the chatbot, where approximate search is acceptable; with them only 33 of 73 lists matched exactly, so evaluation uses exact similarity. Run end to end on the dev split, the Neo4j backend scores F1 0.452 (ATA 0.433, VCA 0.465) against 0.466 in memory, a one-question difference on VCA; the in-memory implementation remains the reference for the reported scores.

### Reasoning layer: snapshot-derived rules

| Rule | Inference | How SGS uses it | Dev evidence |
| --- | --- | --- | --- |
| R1 Domain | Mobile if platforms include Android or iOS, ICS if the ID starts with T0, else Enterprise | platforms shown on ATA cards; never a filter | an Enterprise-only filter raised ATA recall@5 0.57 → 0.60 but assumes the answer space, so it was dropped |
| R2 Retired ID | an Enterprise parent-level ID with the same title as an Enterprise sub-technique → `REPLACED_BY` (103 IDs) | stored; not used to filter | removing retired IDs raised recall@5 to 0.63; reported only as an ablation |
| R3 Hierarchy and tactics | `SUBTECHNIQUE_OF` from the ID pattern, `IN_TACTIC` from `kill_chain_phases` | tactics shown on ATA cards | tactic-first narrowing dropped: Llama-3-8B names the tactic 50% of the time (top 2: 63%), and re-ranking by its guess lowered recall@5 0.57 → 0.50 |
| R4 CAPEC bridge | passage ≈ CAPEC description → `EXPLOITS_WEAKNESS` → CWE | VCA candidates; CAPEC name on VCA cards | alone: VCA recall@1 0.14 vs 0.12 for the baseline |
| R5 CVE examples | passage ≈ CVE description → `HAS_WEAKNESS` → CWE vote | VCA candidates; closest CVE on VCA cards | part of the VCA fusion that lifts recall@5 0.40 → 0.65 |
| R6 Report examples | passage ≈ report-layer behavior from another report → technique vote | ATA candidates; 2 examples on ATA cards | ATA recall@5 0.57 → 0.63 with no filter; 29 of 30 dev gold techniques have an example |

R1–R3 are materialized as properties and edges at load time; R4–R6 run per question as a vector search followed by one-hop aggregation.

### Gold-triple verification (`scripts/verify_triples.py`)

The engine's answers are scored as triples against `data/gold_triples.json`: 69 triples (28 `DESCRIBES_TECHNIQUE`, 41 `DESCRIBES_WEAKNESS`), one per dev-split question, from CTIConnect's expert-verified answers.

1. **Input**: a run's `predictions.jsonl` and the gold file.
2. **Normalize**: extract IDs with CTIConnect's official `extract_ids` / `normalize_id`, so triple scores agree with benchmark scores.
3. **Score**: a predicted triple counts when (report, predicate, ID) equals a gold triple exactly; precision, recall and F1 are reported overall and per predicate.
4. **Diagnostics**: near misses (the predicted technique is the gold's parent or child), retired IDs and IDs absent from the snapshot are counted separately, never as hits.

## 5. Featurization, Embedding & Hybrid Indexing

Candidate generation is the first bottleneck: the small model cannot choose an answer it never sees. Fusing snapshot-derived rankings raises how often the answer reaches the top 5 from 0.57 to 0.63 for ATA and from 0.40 to 0.65 for VCA on the dev split, with no LLM calls and no filtering of the answer space. Configurations were chosen on dev and are confirmed once on the test split.

### Node featurization

Each searchable node gets one text view from its own snapshot fields, embedded with text-embedding-3-large (E, 3,072 dimensions) and L2-normalized:

| Node | Text view |
| --- | --- |
| AttackTechnique (1,076) | title + `contents` (tactics, platforms, detection advice) |
| CWE (1,342) | title + `contents` (consequences, modes of introduction, mitigations) |
| CAPEC (615) | name + `Description` |
| CVE (2,751) | description |
| Report behavior (3,512) | the extracted behavior phrase |

### Scoring functions

With e(x) = E(x) / ‖E(x)‖, a dense ranking orders nodes v by s(q, v) = e(q) · e(v). The three bridges score a target t by the similarity of the records that point to it:

```latex
S_{\mathrm{B}}(t) = \sum_{x \,\in\, \mathrm{top}_{n_B}(\mathrm{B},\, q),\ x \to t} s(q, x), \qquad B \in \{\mathrm{CAPEC}\ (n=10),\ \mathrm{CVE}\ (n=20),\ \mathrm{Report}\ (n=30)\}
```

Report behaviors from the question's own source report are excluded. Rankings are fused by reciprocal-rank fusion:

```latex
\mathrm{RRF}(v) = \sum_{r \in R} \frac{1}{60 + \mathrm{rank}_r(v)}
```

### Queries and chosen configurations

Two queries are built per question: the full question q\_Q and its quoted passage q\_P. The passage alone retrieves worse than the full question, so the full question stays in every configuration.

- **ATA**: RRF of \[RRF(dense(q\_Q), dense(q\_P)) and S\_Report(q\_P)\] over all 1,076 techniques.
- **VCA**: RRF of dense(q\_Q) over CWE records, S\_CAPEC(q\_P) and S\_CVE(q\_P).

&#91;embedded content: runs/snapshot\_recall\_\_dev.jsonl · 73 dev questions · no LLM calls · Oct 16, 2026\]

### Each component alone (dev, recall@5)

| Ranking | ATA (all 1,076 techniques) | VCA |
| --- | --- | --- |
| dense, full question | 0.57 | 0.40 |
| dense, passage | 0.47 | 0.21 |
| report-example vote | 0.53 | — |
| CAPEC bridge | 0.03 | 0.35 |
| CVE vote | — | 0.26 |
| BM25, passage | 0.30 | 0.16 |

**Tested and dropped.** BM25 keyword search and, for ATA, the CAPEC bridge (only 177 CAPECs map to ATT&CK) lowered fused recall. Domain and retired-ID filters raised ATA recall@5 to 0.67 but encode an assumption about the answers. Re-ranking VCA candidates by their single closest CVE example fell to recall@5 0.21.

### Hyperparameters

| Parameter | Value | Set on |
| --- | --- | --- |
| Embedding model | text-embedding-3-large, 3,072 dims, L2-normalized | fixed, matches CTIConnect |
| RRF constant | 60 | fixed, standard |
| Neighbours: CAPEC / CVE / report behaviors | 10 / 20 / 30 | dev split |
| Candidates shown to the model | 5 | dev split: 10 candidates lowered selection accuracy when the gold was shown (ATA 0.50 vs 0.59, VCA 0.43 vs 0.54) |

## 6. Attribution QA Engine & Prompt Templates

The method, Snapshot-Guided Selection (SGS), gathers 5 candidates and their evidence from the snapshot graph and lets the small model make one constrained choice. On the dev split it scores 0.47 F1 with Llama-3-8B (ATA 0.43, VCA 0.49) against 0.26 for vanilla RAG on the same 73 questions. The packaged method reproduces them exactly through the official scorer (F1 0.466; all 73 pick sequences identical to the diagnostic runs). The configuration was chosen on these questions, so the figures are optimistic; the held-out test split will give the unbiased estimate in follow-up work.

### Baselines

| Method | Steps | Source |
| --- | --- | --- |
| CB: closed-book | question → answer | CTIConnect `closed_book`, prompts verbatim |
| VR: vanilla RAG | embed full question → top-5 raw KB records (600 characters each) → answer | CTIConnect `vanilla_rag`, prompts verbatim |
| DtR (reference row) | decompose → retrieve per behavior → aggregate | CTIConnect released code; not reproducible as described in the paper (Section 7) |

The VR answer prompt, used unchanged:

```text
Question:
{question}

Retrieved knowledge-base candidates:
{candidates}

Based on the candidates above, answer the question. Cite the single best-matching
identifier explicitly.
```

### SGS procedure (frozen)

1. **Parse.** Extract the quoted passage and any CVE, CWE or ATT&CK IDs by regex.
2. **Candidates.** Run the chosen configuration of Section 5 and keep the top 5.
3. **Evidence cards.** One line per candidate, built only from the snapshot:
   - ATA: `ID | title | tactics | platforms | detection: <first sentence> | report examples: "<snippet>"; "<snippet>"`, the two report-layer sentences closest to the passage, never from its own report.
   - VCA: `ID | title | consequences | related attack pattern: <CAPEC name> | example vulnerability: <closest CVE description>`.
4. **Choice.** The model returns one ID; JSON-schema structured output restricts it to an enum of the 5 candidate IDs, so an invented ID is impossible.
   - ATA: one call, candidates in retrieval order.
   - VCA: five calls over the cyclic rotations of the 5 candidates (each appears once in every position), majority vote, ties to the better retrieval rank. This cancels position bias: the vote raised VCA from 0.35 to 0.49.
5. **Answer.** Exactly one ID plus its title. A final check confirms the ID is one of the listed snapshot IDs and the title contains no other identifier, since the official scorer counts every ID in the text.

### Prompt (exact)

System: `You are a cyber threat intelligence analyst.` Output schema: `{"id": enum of the candidate IDs}`.

```text
You are mapping a threat-report passage to {MITRE ATT&CK techniques | CWE weaknesses}.

Passage:
"{passage}"

Candidates:
{cards}

Choose the ONE candidate that best describes the behavior in the passage. Judge by
what each candidate means, not by shared words. Return JSON with its id.
```

### Measured ablation (dev, Llama-3-8B fp16; 30 ATA, 43 VCA)

| Configuration | ATA | VCA | Overall | Calls per question |
| --- | --- | --- | --- | --- |
| CB | 0.03 | 0.13 | 0.09 | 1 |
| VR | 0.29 | 0.23 | 0.26 | 1 |
| Snapshot candidates + cards, one listwise call | 0.33 | 0.35 | 0.34 | 1 |
| + report examples in ATA candidates and cards | 0.43 | 0.35 | 0.38 | 1 |
| + VCA 5-rotation vote = **SGS** | **0.43** | **0.49** | **0.47** | ATA 1, VCA 5 |

**Tested and dropped for the choice step.**

- 10 candidates instead of 5: lower accuracy when the gold was shown.
- Pairwise tournament (both orders): ATA 0.40 without examples, 0.37 with; VCA 0.40; up to 8 calls.
- ATA rotation vote: 0.27 without examples, 0.37–0.40 with them; no better than one call.
- Tactic-first narrowing: see R3 in Section 4.
- A fact-checking layer on the output: once candidates are shown, the model's ID and title contradict the graph in 1 of 12 pairs (closed-book: 11 of 21), so the enum restriction and the final check cover it.

On these sample sizes one question moves ATA by 0.03 and VCA by 0.02, so differences of one or two questions between configurations are not meaningful.

## 7. Evaluation Pipeline & Benchmark Protocol

All methods are scored with CTIConnect's official identifier F1 on its 379 attribution questions (ATA 160, VCA 219), with the same model, embedder and data, so differences come from the method alone. This milestone reports results on the 73-question dev split, the same questions used to choose the configuration; the 306-question test split is held out, untouched, for a follow-up evaluation.

### Data and splits

- **Questions**: CTIConnect `data/entity_attribution`, pinned to commit 554797d. Every question has exactly one gold ID.
- **Split by report**: questions are split by source report (seed 13), so no report's passages appear on both sides: dev 73 questions from 29 reports (30 ATA, 43 VCA), test 306.
- **Quick-look set** (`eval_queries.json`): 20 dev questions (10 ATA, 10 VCA), for fast checks and the human evaluation.
- **Known ceiling**: two pairs of ATA questions are word-for-word identical but have different gold IDs (e.g. ata-106 T1105, ata-107 T1059), so maximum ATA F1 is 0.988.

### Automated metrics

| Metric | What it measures | Implementation |
| --- | --- | --- |
| Identifier P / R / F1, exact match | the benchmark score | CTIConnect `evaluation/metrics.py`, vendored unmodified; its 21 tests pass |
| Candidate recall@k (k = 1, 5, 10, 20, 50) | whether the gold ID reaches the model | `scripts/retrieval_recall.py` |
| IDs per answer | over-prediction, which the scorer penalizes | run summary |
| Triple P / R / F1 | `DESCRIBES_*` triples vs 69 gold triples | `scripts/verify_triples.py` (Section 4) |
| RAGAS faithfulness, answer relevancy, context recall | whether the explanation is supported by the retrieved KB entries | RAGAS 0.4 on the 20 eval queries; contexts = what the model was shown, reference = CTIConnect's reference answer |
| Latency, tokens | cost per question | logged per call |

Scoring rule: a failed call is scored as an empty answer (F1 = 0), never dropped. CTIConnect's own aggregator drops missing predictions, which would inflate scores.

### Models and comparison

- **Answering model**: Llama-3-8B-Instruct only, the smallest model in CTIConnect's Table III, served by Ollama at full precision (fp16), temperature 0. Restricting the study to one model keeps every comparison on the same footing within the project's time.
- **Main comparison**: CB vs VR vs SGS, plus the ablation ladder of Section 6.
- **DtR as a reference row only.** The paper describes DtR as decompose → canonicalize → retrieve, but the released code has no canonicalization step and the paper publishes no prompts, so its 0.28 / 0.40 cannot be reproduced. Our paper-faithful reimplementation, with our own canonicalization prompt, scored below the released code on the 20-question set.
- **Significance**: paired bootstrap 95% confidence intervals (10,000 resamples) and McNemar's exact test on exact match, on the dev split now and on the test split in the follow-up.

### Preliminary results (20 questions)

&#91;embedded content: runs/mini20\_\_\* · 20 dev questions · Oct 16, 2026 · paper values from CTIConnect Table III (LLaMA-3-8B)\]

Full precision is used for all later runs. DtR's shortfall is structural, not quantization: its recall (0.40) equals vanilla RAG's, but it outputs about 3 IDs per answer against 1.6, so precision falls to 0.13. Vanilla RAG at full precision (ATA 0.37, VCA 0.30) matches or exceeds the paper's 0.28 / 0.22. With 20 questions, one answer moves F1 by 0.05, so these numbers guide design choices only; claims come from the test split.

### Development-set results (73 questions, Llama-3-8B fp16)

SGS reaches F1 0.466 against 0.256 for vanilla RAG on the same questions, a paired gain of +0.21 that holds up under resampling; the gain is clear for VCA and not yet resolved for ATA's 30 questions.

| Method | ATA (n = 30) | VCA (n = 43) | Overall |
| --- | --- | --- | --- |
| CB, closed-book | 0.033 | 0.132 | 0.091 |
| VR, vanilla RAG | 0.289 | 0.233 | 0.256 |
| **SGS** | **0.433** | **0.488** | **0.466** |
| CTIConnect reported, LLaMA-3-8B (all 379 questions) | VR 0.28, DtR 0.28 | VR 0.22, DtR 0.40 | — |

| Paired comparison | Difference in F1 | 95% bootstrap CI |
| --- | --- | --- |
| SGS − VR, overall | +0.210 | \[+0.082, +0.338\] |
| SGS − VR, ATA | +0.144 | \[−0.067, +0.356\] |
| SGS − VR, VCA | +0.256 | \[+0.093, +0.419\] |
| VR − CB, overall | +0.164 | \[+0.050, +0.279\] |

On exact match SGS is right where VR is wrong on 24 questions and the reverse on 5 (McNemar exact p = 0.0005). Two caveats bound these numbers: the configuration was chosen on these questions, so they are optimistic, and the paper's figures cover all 379 questions rather than this subset. The held-out test split addresses both in follow-up work.

### RAGAS on the 20 eval queries

SGS, with its explanation step, scores higher than vanilla RAG on all three RAGAS metrics, with modest margins on 20 queries. The explanation step adds no identifier, so official F1 is unchanged (checked per query).

| Method | F1 | Faithfulness | Answer relevancy | Context recall |
| --- | --- | --- | --- | --- |
| VR, vanilla RAG | 0.33 | 0.46 | 0.60 | 0.39 |
| **SGS + explanation** | **0.45** | **0.51** | **0.67** | **0.46** |

Setup: `scripts/ragas_eval.py`, RAGAS 0.4.3 collections metrics, judge gpt-5.4-mini-2026-03-17, embeddings text-embedding-3-large. Contexts are exactly what each method showed the model (SGS: the 5 evidence cards; VR: the 5 KB records at 600 characters). References are CTIConnect's reference answers, which also cover the questions' extra asks (likelihood of exploit, affected platforms, consequences). Judge cost: 120 calls and $0.09 for the answer-relevancy pass.

Reading the numbers: faithfulness around 0.5 means about half of each explanation's claims are not stated in the cards. The model fills the questions' extra asks from its own knowledge, which the evidence cards do not cover. Context recall below 0.5 has the same cause: the cards support the ID, not every detail of the reference answer. Adding consequences and platforms to the cards is the obvious next step for explanation quality.

### Human evaluation protocol

The author scores each system's answer on the 20 eval queries on a 1–5 scale, with answers shuffled and system labels hidden:

| Score | Factuality | Completeness | Coherence |
| --- | --- | --- | --- |
| 5 | ID and every claim match the KB entry and the passage | names the ID, explains the mapping, answers every sub-question | reasoning follows from passage to entry with no gaps |
| 4 | correct ID, one minor inaccuracy | one sub-question thin | one small gap |
| 3 | related ID (parent, child or sibling) or one wrong claim | one sub-question missing | understandable but loosely argued |
| 2 | wrong ID with a plausible argument | several parts missing | hard to follow |
| 1 | wrong ID or invented content | does not answer | incoherent |

As an individual project there is one rater; to check consistency, the author re-scores a random 10 answers a week later and reports quadratic-weighted Cohen's κ between the two passes.

## 8. Software Interfaces & Data Models

The engine is a Python package (`ctiattr`) behind a FastAPI service; a Streamlit chatbot calls it over HTTP, and the evaluation runner calls the same engine in-process so benchmark and chatbot answers come from one code path.

### Data models (`schemas.py`, Pydantic v2)

```python
# src/ctiattr/schemas.py (excerpt: graph, report layer and API models)
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field

ID_PATTERN = r"^(T\d{4}(\.\d{3})?|CWE-\d+|CAPEC-\d+|CVE-\d{4}-\d{4,})$"

class Taxonomy(str, Enum):
    ATTACK = "attack"   # ATA: MITRE ATT&CK techniques
    CWE = "cwe"         # VCA: CWE weaknesses

class NodeLabel(str, Enum):
    TECHNIQUE = "AttackTechnique"
    CWE = "CWE"
    CAPEC = "CAPEC"
    CVE = "CVE"

class KBNode(BaseModel):
    id: str = Field(..., pattern=ID_PATTERN)
    label: NodeLabel
    title: str
    text: str = ""                      # snapshot contents, CAPEC or CVE description
    domain: str | None = None           # ATT&CK only (rule R1): enterprise | mobile | ics
    replaced_by: str | None = None      # ATT&CK only (rule R2)
    tactics: list[str] = Field(default_factory=list)

class Predicate(str, Enum):
    SUBTECHNIQUE_OF = "SUBTECHNIQUE_OF"; IN_TACTIC = "IN_TACTIC"; REPLACED_BY = "REPLACED_BY"
    EXPLOITS_WEAKNESS = "EXPLOITS_WEAKNESS"; MAPS_TO_TECHNIQUE = "MAPS_TO_TECHNIQUE"
    HAS_WEAKNESS = "HAS_WEAKNESS"; GROUNDS = "GROUNDS"
    DESCRIBES_TECHNIQUE = "DESCRIBES_TECHNIQUE"; DESCRIBES_WEAKNESS = "DESCRIBES_WEAKNESS"

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    predicate: Predicate
    origin: str                         # "corpus_kb", "report_layer" or "rule:R<n>"

class ReportChunk(BaseModel):
    chunk_id: str = Field(..., pattern=r"^BLOG-\d+#\d+$")
    report_id: str = Field(..., pattern=r"^BLOG-\d+$")
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)
    text: str

class GroundedBehavior(BaseModel):
    chunk_id: str
    report_id: str
    behavior: str = Field(..., min_length=1)   # verbatim substring of the chunk
    technique_id: str = Field(..., pattern=r"^T\d{4}(\.\d{3})?$")
    technique_name: str = ""

class GoldTriple(BaseModel):
    id: str
    subject_id: str = Field(..., pattern=r"^BLOG-\d+$")
    predicate: Predicate
    object_id: str = Field(..., pattern=ID_PATTERN)
    evidence_text: str
    source_question: str

class Candidate(BaseModel):
    id: str = Field(..., pattern=ID_PATTERN)
    rank: int = Field(..., ge=1)
    card: str                           # evidence card shown to the model
    sources: list[str] = Field(default_factory=list)   # "dense_question", "capec", "cve", "report"

class LLMCall(BaseModel):
    step: str                           # "select" or "explain"
    prompt: str
    output: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float

class AttributionRequest(BaseModel):
    passage: str = Field(..., min_length=10)
    taxonomy: Taxonomy
    question: str | None = None         # defaults to the passage
    report_id: str | None = None        # excluded from report-layer examples
    method: str = "sgs"                 # "cb" | "vr" | "sgs"
    explain: bool = True

class AttributionResponse(BaseModel):
    answer_id: str = Field(..., pattern=ID_PATTERN)   # exactly one identifier
    answer_title: str
    explanation: str = ""               # no identifier other than answer_id
    candidates: list[Candidate] = Field(default_factory=list)
    votes: list[str] = Field(default_factory=list)    # VCA: the pick of each rotation
    calls: list[LLMCall] = Field(default_factory=list)
    model: str
    latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

### API endpoints (FastAPI)

| Method | Route | Request → response | Used by |
| --- | --- | --- | --- |
| POST | `/attribute` | `AttributionRequest` → `AttributionResponse` | chatbot, evaluation runner |
| GET | `/kb/{node_id}` | → `KBNode` with parent and children | chatbot "explain this entry" panel |
| GET | `/kb/search?q=&taxonomy=&k=` | → `list[Candidate]` | chatbot search box, retrieval diagnostics |
| GET | `/health` | → model name, Ollama digest, graph counts | deployment checks |

### Frontend

The Streamlit page takes a pasted passage, shows the chosen ID with its KB definition, the graph path drawn as a small tree (tactic → technique → sub-technique, or passage → CAPEC/CVE → CWE), and an expandable list of the candidates and LLM calls behind the answer.

## 9. Risks, Status & References

### Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Retrieval caps every method (gold in top 5 for 57% / 40% of dev questions) | snapshot candidates (CAPEC, CVE and report-example bridges, RRF): dev recall@5 ATA 0.63, VCA 0.65; recall@k reported before any F1 |
| Small-sample noise: on 20 questions one answer moves F1 by 0.05–0.10 | decisions made on the 73-question dev split; final claims only from the 306-question test split with bootstrap CIs |
| Choices tuned on 73 dev questions overfit them | results labelled as dev results, with per-component recall and bootstrap CIs; the test split stays held out for the follow-up |
| Baseline drift from the paper (quantization, decoding defaults) | fp16 models; 4-bit vs fp16 compared on the same 20 questions; differences reported, not hidden |
| Overfitting prompts to dev questions | test split frozen and run once |
| DGX Ollama instability (stalled queue observed on Oct 16) | resumable runner; the Mac's Ollama serves the same model digest as a fallback |

### Repository checkpoint (branch `setup/baselines`)

- [x] Pinned CTIConnect data fetch with checksum verification (`scripts/fetch_cticonnect.py`)
- [x] Official scorer vendored, its 21 tests passing
- [x] CB, VR, DtR (released code) and paper-faithful DtR baselines, resumable runner with full traces
- [x] Frozen dev/test split by report; `data/eval_queries.json` (20) and `data/gold_triples.json` (69)
- [x] Retrieval recall diagnostic (`scripts/retrieval_recall.py`)
- [x] `schema.cypher` and the Neo4j loader (scripts/load\_graph.py; Cypher bridges via --backend neo4j, dev F1 0.452)
- [x] Snapshot-derived rules R1–R6, candidate generators and report layer (src/ctiattr/snapshot.py, candidates.py, report\_layer.py)
- [x] SGS engine (done; dev F1 0.466); test-split run on the 306 held-out questions deferred to follow-up work
- [x] Triple verification (`scripts/verify_triples.py`; dev triple F1: SGS 0.511, VR 0.253), Pydantic models (`src/ctiattr/schemas.py`), 36 passing tests
- [x] Explanation step with ID post-filter and RAGAS evaluation (`scripts/ragas_eval.py`)
- [ ] FastAPI service and Streamlit chatbot

### References

- Boyer, A., Dogdu, E., Choupani, R., Watson, J. S., Sanchez, D., & Ametu, A. (2024). Toward a unified cybersecurity knowledge graph: Leveraging ontologies and open data sources. *Recent Advances in Next-Generation Data Science*, Springer.
- Cheng, Y., Liu, Y., Li, C., Song, D., & Gao, P. (2026). CTIConnect: A benchmark for retrieval-augmented LLMs over heterogeneous cyber threat intelligence. *KDD '26*. [arXiv:2510.11974](https://arxiv.org/abs/2510.11974); [code and data](https://github.com/peng-gao-lab/CTIConnect)
- Cheng, Y., Li, C., Cui, Q., Wang, L., Chen, Y., Ding, W., & Gao, P. (2026). CTIFoundry: An agent-native corpus scaffold for cyber threat intelligence. [arXiv:2608.18613](https://arxiv.org/abs/2608.18613)
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR '09*.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *EACL 2024 (demos)*.
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 33*.
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*.
