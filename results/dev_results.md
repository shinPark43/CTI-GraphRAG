# Development-split results (for the final report)

Results moved out of the Milestone 2 design document, which now holds design only.
The full document as it stood with all results (rev 81, 2026-10-17) is kept at
`reports/design_doc_rev81_with_results.md`: Sections 3–7 there carry the recall
tables, the ablation ladder, dev F1 with bootstrap CIs and McNemar, and RAGAS.
Everything below is on the 73-question dev split (30 ATA, 43 VCA), which was also
used to choose configurations, so all numbers are optimistic. The 306-question
test split is untouched.

## Headline (Llama-3-8B-Instruct fp16, official CTIConnect scorer)

| Method | ATA | VCA | Overall | Run |
| --- | --- | --- | --- | --- |
| CB, closed-book | 0.033 | 0.132 | 0.091 | `runs/dev__closed_book__llama3-8b-fp16` |
| VR, vanilla RAG | 0.289 | 0.233 | 0.256 | `runs/dev__vanilla_rag__llama3-8b-fp16` |
| SGS | 0.433 | 0.488 | 0.466 | `runs/dev__sgs__llama3-8b-fp16` |
| SGS, Neo4j backend | 0.433 | 0.465 | 0.452 | `runs/dev__sgs-neo4j__llama3-8b-fp16` |

Paired bootstrap SGS − VR: +0.210 [+0.082, +0.338]; ATA +0.144 [−0.067, +0.356];
VCA +0.256 [+0.093, +0.419]. McNemar on exact match: 24 vs 5 discordant, p = 0.0005.
Triple F1 (69 gold triples): SGS 0.511, VR 0.253, CB 0.130.
RAGAS, 20 eval queries, judge gpt-5.4-mini: faithfulness / answer relevancy /
context recall = SGS 0.51 / 0.67 / 0.46, VR 0.46 / 0.60 / 0.39.

## Graph embeddings, stage 1: candidate recall@5 (no LLM)

`scripts/graph_embed_recall.py` → `runs/graph_embed_recall__dev.json` (89 rows, per-question ranks).
A = propagated text embeddings; B = text + node2vec (corpus_kb graph only, anchored on 10 nearest nodes).
Own-report behaviors always excluded.

| Ranking | ATA | VCA |
| --- | --- | --- |
| Text embedding, question (what VR retrieves) | 0.57 | 0.40 |
| SGS fused candidates (frozen) | 0.63 | 0.65 |
| A, kb+report, α = 0.4, K = 1, question + passage (shared setting) | 0.80 | 0.63 |
| A, best per task (ATA: report edges, α 0.4, K 2; VCA: CAPEC+CVE edges, α 0.6, K 1, question) | 0.80 | 0.70 |
| B, best (node2vec q = 2, λ = 0.1) | 0.60 | 0.56 |
| SGS + A (RRF) | 0.80 | 0.67 |

Sensitivity: ATA with report edges stays at 0.67–0.80 over all six (α, K); taxonomy-only
edges give 0.47–0.57; kb+report at K = 2 falls to 0.43 (tactic hubs). VCA needs CAPEC/CVE
edges and K = 1 (0.56–0.70); B loses VCA recall as λ grows (0.56 → 0.26).
Shared setting vs text alone: ATA +7/−0 questions, CI [+0.10, +0.40]; VCA +10/−0,
CI [+0.12, +0.37]. Vs SGS candidates: ATA +5/−0, CI [+0.03, +0.30]; VCA +4/−5.

Stage 2 (Llama runs of GE-RAG and SGS with A's candidates) is deferred to the final report.

## Graph Data Science on the corpus_kb graph

`gds.cypher` via `scripts/gds_analysis.py` → `runs/gds_analysis.json`. Neo4j GDS 2.13.14;
projection = corpus_kb only (5,803 nodes, 6,764 edges; report layer excluded), all
algorithms run in under 1 s. No questions used.

- **Degree.** Tactics are hubs (mean 72, max 304: defense-evasion). CVEs have 1–4 edges.
  891 of 1,342 CWEs (66%) and 131 of 615 CAPECs are isolated: the snapshot carries no CWE
  hierarchy, so a CWE is connected only if a CVE or CAPEC points at it.
- **WCC.** 1,116 components; one giant component of 4,451 nodes (1,066 techniques, 344 CWEs,
  459 CAPECs, 2,565 CVEs). Most other components are singletons.
- **PageRank** (stored direction, d = 0.85, converged in 4 iterations). CWE leaders follow how
  often CVEs are labelled with them: CWE-79 (37.7), CWE-89, CWE-74, CWE-200, CWE-787.
  Technique leaders: T1195, T1027, T1036, T1218. CAPEC nodes have only outgoing edges, so their
  PageRank is flat; betweenness ranks them instead.
- **Betweenness.** Bridges between the vulnerability and ATT&CK sides: CWE-20, CWE-79, CWE-200
  and CAPEC-13, CAPEC-60, CAPEC-76.
- **Louvain** (single thread for reproducibility): 1,149 communities, modularity 0.864 over 4 levels.
  The largest group CVEs around one CWE family (XSS 384, input validation 361, SQLi 278,
  memory reads 231) or techniques around tactics (persistence / priv-esc 327, defense evasion 228).
  79.4% of techniques sit in a community whose dominant tactic is one of their own.
- **Bridge coverage.** 260 CWEs have CVEs, 336 have CAPECs, 149 reach a technique through CAPEC;
  189 techniques have a CAPEC. 7,866 CVE → CWE ← CAPEC → technique paths.
  Example (Dijkstra, 3 hops): CVE-2020-12790 → CWE-74 → CAPEC-13 → T1562.003.

### GDS outputs as retrieval signals, stage 1 (recall, no LLM)

`scripts/gds_recall.py` → `runs/gds_recall__dev.json`. PR = cos + β·(log PageRank scaled to
[0, 1] within the target kind); community = cos + γ·(share of the question's 10 nearest KB nodes,
cosine-weighted, in the candidate's Louvain community). β, γ picked on dev (optimistic).

| Ranking (question + passage) | ATA @1 | ATA @5 | VCA @1 | VCA @5 | VCA @20 |
| --- | --- | --- | --- | --- | --- |
| Text embedding | 0.33 | 0.57 | 0.12 | 0.37 | 0.70 |
| PR prior (ATA β 0.05, VCA β 0.1) | 0.30 | 0.60 | 0.23 | 0.49 | 0.79 |
| Community vote (ATA γ 0.02, VCA γ 0.2) | 0.33 | 0.53 | 0.23 | 0.49 | 0.67 |
| PR + community | 0.30 | 0.60 | 0.28 | 0.58 | 0.81 |
| SGS fused candidates (frozen) | 0.33 | 0.63 | 0.16 | 0.65 | 0.74 |
| SGS + PR (RRF) | 0.43 | 0.77 | 0.35 | 0.63 | 0.81 |
| SGS + PR + community (RRF) | 0.40 | 0.77 | 0.33 | 0.65 | 0.81 |

Where the gold answers sit: every dev gold CWE and technique is in the connected graph, though
66% of all CWEs are isolated; the median gold CWE is at the 95.5th PageRank percentile (techniques:
70th); the gold node is in one of the question's anchor communities for 83% of ATA and 65% of VCA.
So PageRank works as a base-rate prior for VCA, learned from the knowledge base's own CVE labels.
Leak check: no dev VCA passage is a near-duplicate of a corpus_kb CVE (median best cosine 0.48,
none > 0.9) and none names a KB CVE. Compared with graph embedding A (ATA 0.80, VCA 0.70 @5), GDS
signals mostly sharpen the top of the list (VCA @1 0.16 → 0.35 with SGS).

## Edge provenance and extraction confidence

`src/ctiattr/provenance.py`, written by `scripts/load_graph.py` (`--annotate-only` re-runs just this step).
Every relationship carries `source` (a `:Source` node: the CTIConnect snapshot at its pinned commit,
snapshot date 2025-09-01, or the report-layer build), `method`, `derivation`, `loaded_at` and `confidence`.

| Method | Relationships | Confidence |
| --- | --- | --- |
| asserted (stated in a corpus_kb record) | IN_TACTIC 1,374, EXPLOITS_WEAKNESS 1,212, MAPS_TO_TECHNIQUE 271, HAS_WEAKNESS 3,286 | 1.0 |
| derived (rule over corpus_kb) | SUBTECHNIQUE_OF 518, REPLACED_BY 103 | 1.0 |
| segmented (report chunking) | HAS_CHUNK 650, HAS_BEHAVIOR 3,512 | 1.0 |
| extracted (LLM, gpt-5.4-mini, 2026-10-17) | GROUNDS 3,512 | audit precision of its tier |

Extra edge properties: HAS_WEAKNESS has NVD `weakness_type` (1,882 Primary, 1,404 Secondary);
MAPS_TO_TECHNIQUE has CAPEC's own `capec_entry_name` and its agreement with the current ATT&CK
title (1 of 271 looks stale: CAPEC-662 "Man in the Browser" → T1185 "Browser Session Hijacking");
HAS_BEHAVIOR and ReportBehavior have the behavior's character span in the report (3,512 of 3,512
resolve to the exact text).

**Extraction confidence.** Each GROUNDS edge passed the hard checks (ID in the snapshot, phrase
verbatim in its chunk; 6 unknown-ID and 223 non-verbatim records were dropped of 3,741). Two soft
signals sort it into a tier: agreement between the name the model wrote and the official title
of the ID it gave (low = likely wrong ID), and the technique's rank for the behavior phrase alone
by embedding (high = meaning comes from context). Tier precision from a stratified hand audit
(`data/derived/extraction_audit.csv`, 25 per tier, seed 6338; first-pass labels by Claude,
to be checked by the author):

| Tier | Rule | Edges | Precision (Wilson 95%) |
| --- | --- | --- | --- |
| supported | name agreement ≥ 0.6, rank ≤ 50 | 2,488 (70.8%) | 0.76 [0.57, 0.89] (19/25) |
| context | name agreement ≥ 0.6, rank > 50 | 934 (26.6%) | 0.24 [0.12, 0.43] (6/25) |
| conflict | name agreement < 0.6 | 90 (2.6%) | 0.32 [0.17, 0.52] (8/25) |

Estimated report-layer precision, tier-weighted: **0.61**. Typical errors: a sibling sub-technique
(HTML smuggling → T1027.016 instead of .006; Java → T1059.006 Python), vague phrases mapped to a
technique ("initial access" → T1190), and defender actions taken as attacker behavior. Conflicts
are partly ATT&CK renames (T1574.001 is now "DLL"; extortion is inside T1657), so name
disagreement alone is not an error. SGS still uses every edge unweighted; confidence-weighted
votes are a follow-up.

## Schema-constrained extraction (Pydantic)

`ExtractionReply` / `ExtractedTTP` / `GroundedBehavior` in `src/ctiattr/schemas.py`. Generation is
constrained to the strict JSON schema derived from `ExtractionReply` (`report_layer.reply_schema()`;
accepted by both OpenAI strict mode and Ollama). Every record is then validated twice: shape and ID
format (`ExtractedTTP`), then grounding with the chunk as validation context (`GroundedBehavior`:
ID in the snapshot → `unknown_id`, phrase verbatim → `not_verbatim`). The committed gpt-5.4-mini layer
was produced in JSON mode; re-validating its 3,741 cached records with the Pydantic path reproduces
`report_ttps.jsonl` byte for byte (0 schema, 6 unknown-ID, 223 not-verbatim drops; 6.1% dropped).
The answer step's constrained selection is likewise re-checked by a Pydantic `Literal` model of the
listed candidates.

Local model, same prompt and schema (`scripts/compare_extractors.py`, 40 chunks, seed 6338,
llama3:8b-instruct-fp16 on the local Ollama, temperature 0):

| | Llama-3-8B (local) | gpt-5.4-mini (committed) |
| --- | --- | --- |
| Records returned / kept | 209 / 148 | — / 239 |
| Schema failures | 0 | 0 |
| Dropped: unknown ID / not verbatim | 3 / 58 (29.2% dropped) | 6.1% dropped (full build) |
| Latency per chunk | 12.3 s mean, 23.8 s max | — |
| Agreement on (chunk, technique), exact | P 0.18, R 0.11, F1 0.14 vs gpt | |
| Agreement, technique family | P 0.29, R 0.19, F1 0.23 vs gpt | |

Constrained decoding removes malformed output entirely, but the 8B model paraphrases instead of
quoting (58 non-verbatim) and agrees with the larger model on few techniques, so the committed layer
stays the gpt-5.4-mini build; Llama-3-8B stays the answering model.

## Graph citations and latency

`src/ctiattr/citations.py`; `SGS.answer` adds `citations` and a `latency_s` breakdown to the trace
(never to the scored answer text). Existing runs are cited post hoc, without LLM calls, by
`scripts/cite_predictions.py <run>`, which also runs every citation's Cypher against Neo4j.

Citation kinds: **report** (ThreatReport → ReportChunk → ReportBehavior → technique, with the
behavior's character span, tier and GROUNDS confidence; never the question's own report),
**capec** (CAPEC → technique / CWE, for CAPEC patterns similar to the passage), **cve**
(CVE → CWE with NVD Primary/Secondary), **taxonomy** (sub-technique and tactic edges). Each records
whether the model saw it on the candidate card.

Dev SGS run (`runs/dev__sgs__llama3-8b-fp16`, 73 answers): 249 citations (report 79, CAPEC 66,
CVE 58, taxonomy 46); every answer has at least one evidence path; **249/249 paths verified in
Neo4j**. Correct and wrong answers carry about the same number of evidence paths (2.71 vs 2.85),
so path count alone does not signal correctness; the citations make an answer checkable, not right.
Re-running two dev questions live gives the identical answer text with citations attached.

Latency (recorded run, Llama-3-8B fp16 on the local Ollama): ATA 4.0 s mean (1 LLM call),
VCA 15.1 s mean (5 rotation calls, max 17.4 s); the LLM is > 99% of the time. Live: retrieval
≈ 5 ms, citations ≈ 7 ms (0.6 s once, to load the confidence signals).

## Identifier hallucination

`scripts/hallucination_check.py` → `runs/hallucination__dev.json`. IDs are extracted as the official
scorer does and checked against the snapshot. No LLM calls.

| Method | Fabricated ID (answers) | No ID | Hedged (> 1 ID) |
| --- | --- | --- | --- |
| Closed-book | 10% (7 of 79 IDs; ATA 23%, e.g. the nonexistent T1193.001 three times) | 7% | 11% |
| Vanilla RAG | 0% | 0% | 23% (VCA 28%) |
| SGS | 0% | 0% | 0% |
| SGS, Neo4j backend | 0% | 0% | 0% |

Retrieval removes fabricated IDs but not hedging; SGS removes both by construction (the model
picks one of five listed snapshot IDs under a constrained-decoding enum).
