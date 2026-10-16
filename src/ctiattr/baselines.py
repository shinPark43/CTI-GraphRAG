"""CTIConnect's entity-attribution baselines, ported from baselines/{closed_book,
vanilla_rag,dtr}/run.py (MIT). Prompts, token limits, top-k and candidate
truncation are copied verbatim so results are comparable with the paper.

Each method takes (qa, ctx) and returns (final_text, trace). ctx holds
"llm" (ChatLLM), "retriever" (Retriever) and "k".
"""

from __future__ import annotations

from ctiattr.data import QA, TASK_KB
from ctiattr.retrieval import format_candidates

# --- closed book -----------------------------------------------------------

CB_SYSTEM = ("You are a Cyber Threat Intelligence expert. Answer the question "
             "directly and concisely. When the question asks for a CWE, CVE, "
             "CAPEC, or MITRE ATT&CK identifier, state the exact identifier "
             "(e.g., CWE-79, T1059.001) explicitly in your answer.")


def closed_book(qa: QA, ctx: dict) -> tuple[str, dict]:
    call = ctx["llm"].chat(qa.question, system=CB_SYSTEM, max_tokens=512)
    return call.output, {"calls": [call.to_dict()]}


# --- vanilla RAG -----------------------------------------------------------

VR_SYSTEM = ("You are a Cyber Threat Intelligence expert. Use the retrieved "
             "knowledge-base candidates to answer. State the exact identifier "
             "(CWE-/CVE-/CAPEC-/T-) explicitly in your answer.")

VR_PROMPT = """Question:
{question}

Retrieved knowledge-base candidates:
{candidates}

Based on the candidates above, answer the question. Cite the single best-matching
identifier explicitly."""


def vanilla_rag(qa: QA, ctx: dict) -> tuple[str, dict]:
    hits = ctx["retriever"].retrieve(qa.question, TASK_KB[qa.task], k=ctx["k"])
    prompt = VR_PROMPT.format(question=qa.question, candidates=format_candidates(hits))
    call = ctx["llm"].chat(prompt, system=VR_SYSTEM, max_tokens=512)
    return call.output, {
        "retrieved": [[h.doc_id for h in hits]],
        "calls": [call.to_dict()],
    }


# --- decompose-then-retrieve ----------------------------------------------

DTR_DECOMPOSE = """You are a CTI analyst. Decompose the security report
passage in the question below into its distinct atomic behaviors (each a single
actionable security event). Output one behavior per line, no numbering, 2-5
behaviors total. Do not answer the question.

{question}

Atomic behaviors:"""

DTR_SYSTEM = ("You are a CTI expert mapping report behaviors to taxonomy "
              "entries. Use only the retrieved candidates. State the exact "
              "identifier(s) (CWE-/T-) explicitly.")

DTR_VALIDATE = """Original question:
{question}

For each atomic behavior below, the most similar taxonomy candidates were
retrieved. Select the single best-matching identifier per behavior (or none if
no candidate fits), then present the final consolidated answer to the original
question with all selected identifiers stated explicitly.

{behavior_blocks}"""


def dtr(qa: QA, ctx: dict) -> tuple[str, dict]:
    llm, kind = ctx["llm"], TASK_KB[qa.task]
    dec = llm.chat(DTR_DECOMPOSE.format(question=qa.question), max_tokens=256)
    behaviors = [b.strip("-* \t") for b in dec.output.splitlines() if b.strip()][:5]
    if not behaviors:
        behaviors = [qa.question]
    blocks, retrieved = [], []
    for b in behaviors:
        hits = ctx["retriever"].retrieve(b, kind, k=ctx["k"])
        retrieved.append([h.doc_id for h in hits])
        blocks.append(f"Behavior: {b}\nCandidates:\n{format_candidates(hits, max_chars=300)}")
    val = llm.chat(DTR_VALIDATE.format(question=qa.question,
                                       behavior_blocks="\n\n---\n\n".join(blocks)),
                   system=DTR_SYSTEM, max_tokens=700)
    return val.output, {
        "behaviors": behaviors,
        "retrieved": retrieved,
        "calls": [dec.to_dict(), val.to_dict()],
    }


METHODS = {"closed_book": closed_book, "vanilla_rag": vanilla_rag, "dtr": dtr}
