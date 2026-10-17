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
