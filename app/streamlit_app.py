"""CTI attribution chatbot: a threat-report passage in, an ATT&CK technique or CWE out,
with the graph paths behind the answer.

    uv run streamlit run app/streamlit_app.py

Needs the local Ollama (LLM_* in .env) and, for analytics and path checks, Neo4j
(docker compose up -d). Example questions come from the development split only.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ctiattr.config import RUNS
from ctiattr.data import passage as quoted_passage
from ctiattr.schemas import AttributionRequest, AttributionResponse, Taxonomy
from ctiattr.service import AttributionService, dev_examples, guess_taxonomy

st.set_page_config(page_title="CTI Attribution GraphRAG", page_icon="🕸️", layout="wide")

METHODS = {"SGS (knowledge graph)": "sgs", "Vanilla RAG": "vr", "Closed-book": "cb"}
COLORS = {"ThreatReport": "#f4d35e", "ReportChunk": "#f6e7a6", "ReportBehavior": "#ee964b",
          "AttackTechnique": "#7fb7e3", "Tactic": "#b8d8f0", "CWE": "#e88a8a", "CAPEC": "#c3a6e0",
          "CVE": "#9ad1a5"}


@st.cache_resource(show_spinner="Loading indexes and connecting to Neo4j…")
def service() -> AttributionService:
    graph = None
    try:
        from ctiattr.graph_db import GraphDB
        graph = GraphDB()
        graph.run("RETURN 1")
    except Exception:  # noqa: BLE001 - run without analytics
        graph = None
    return AttributionService(graph=graph)


@st.cache_data
def examples() -> list[dict]:
    return [{"id": qa.id, "task": qa.task, "question": qa.question, "passage": qa.passage,
             "report_id": qa.report_id, "gold": qa.gold} for qa in dev_examples()]


@st.cache_data
def gds_summary() -> dict | None:
    path = RUNS / "gds_analysis.json"
    return json.loads(path.read_text()) if path.exists() else None


svc = service()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "verified" not in st.session_state:
    st.session_state.verified = {}

# --- sidebar ------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    method_label = st.radio("Method", list(METHODS), help="SGS answers with graph citations; the "
                                                          "baselines are here for comparison.")
    tax_label = st.selectbox("Taxonomy", ["Auto", "MITRE ATT&CK (ATA)", "CWE (VCA)"])
    explain = st.toggle("Explain the answer", value=True)

    st.header("Examples (dev split)")
    ex = examples()
    pick = st.selectbox("Question", range(len(ex)), format_func=lambda i: (
        f"{ex[i]['id']} · {ex[i]['passage'][:60]}…"))
    ask_example = st.button("Ask this example", width="stretch")

    st.header("Graph")
    if svc.graph is None:
        st.warning("Neo4j unavailable: answering with in-memory indexes; no analytics or path checks.")
    else:
        counts = svc.stats()
        st.caption(f"{sum(counts['nodes'].values()):,} nodes · {sum(counts['relationships'].values()):,} "
                   f"relationships")
        g = gds_summary()
        if g:
            st.caption(f"GDS: {g['louvain']['communityCount']:,} Louvain communities (modularity "
                       f"{g['louvain']['modularity']:.2f}), {g['wcc']['componentCount']:,} components")
    st.caption(f"Model: `{svc.model}`")
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# --- rendering ----------------------------------------------------------------

def path_text(cit: dict) -> str:
    return " ".join(f"**{s['node']}**" if s.get("node") else f"→ `{s['rel']}` →" for s in cit["path"])


def subgraph_dot(resp: dict) -> str:
    lines = ['digraph G {', 'rankdir=LR; node [shape=box, style="rounded,filled", fontname=Helvetica, '
             'fontsize=10]; edge [fontname=Helvetica, fontsize=8, color="#666666"];']
    seen, edges = set(), set()
    for c in resp["citations"]:
        nodes = [s for s in c["path"] if s.get("node")]
        rels = [s["rel"] for s in c["path"] if s.get("rel")]
        for n in nodes:
            if n["node"] not in seen:
                seen.add(n["node"])
                bold = ", penwidth=2.5" if n["node"] == resp["answer_id"] else ""
                lines.append(f'"{n["node"]}" [fillcolor="{COLORS.get(n["label"], "#dddddd")}"{bold}];')
        for a, b, r in zip(nodes, nodes[1:], rels):
            if (a["node"], b["node"], r) not in edges:
                edges.add((a["node"], b["node"], r))
                lines.append(f'"{a["node"]}" -> "{b["node"]}" [label="{r}"];')
    return "\n".join(lines + ["}"])


def render(msg: dict, idx: int) -> None:
    resp = msg["response"]
    gold = msg.get("gold")
    head = f"### {resp['answer_id'] or 'no identifier'}"
    if resp["answer_title"]:
        head += f" — {resp['answer_title']}"
    st.markdown(head)
    if gold:
        ok = (resp["answer_id"] or "").upper() in gold
        st.markdown(f"{'✅' if ok else '❌'} gold answer: `{', '.join(gold)}` · method: {msg['method']}")
    else:
        st.caption(f"method: {msg['method']}")
    if resp["explanation"]:
        st.markdown(resp["explanation"])

    lat = resp.get("latency") or {}
    cols = st.columns(4)
    cols[0].metric("Total", f"{lat.get('total', resp['latency_ms'] / 1000):.1f} s")
    cols[1].metric("LLM", f"{lat.get('llm', 0):.1f} s")
    cols[2].metric("Retrieval", f"{lat['retrieval'] * 1000:.0f} ms" if "retrieval" in lat else "—")
    cols[3].metric("Graph citations", len(resp["citations"]))

    if msg["method"] != "sgs":
        with st.expander("LLM call"):
            for c in resp["calls"]:
                st.code(c["prompt"], language=None)
        return

    tabs = st.tabs(["Citations", "Subgraph", "Candidates", "Graph analytics", "LLM calls"])
    with tabs[0]:
        for j, c in enumerate(resp["citations"]):
            with st.container(border=True):
                meta = [c["kind"], c["method"]]
                if c.get("similarity") is not None:
                    meta.append(f"similarity {c['similarity']:.2f}")
                if c.get("confidence") is not None:
                    meta.append(f"confidence {c['confidence']:.2f}" + (f" ({c['tier']})" if c.get("tier") else ""))
                if c.get("weakness_type"):
                    meta.append(f"NVD {c['weakness_type']}")
                if c.get("shown_to_model"):
                    meta.append("shown to the model")
                st.caption(" · ".join(meta))
                st.markdown(path_text(c))
                if c.get("evidence"):
                    st.markdown(f"> {c['evidence']}")
                if c.get("span"):
                    st.caption(f"report {c['span'][0]}, characters {c['span'][1]}–{c['span'][2]}")
                st.code(c["cypher"], language="cypher")
                key = f"{idx}-{j}"
                if svc.graph is not None and st.button("Verify in Neo4j", key=f"verify-{key}"):
                    from ctiattr.schemas import PathStep
                    st.session_state.verified[key] = svc.verify([PathStep(**s) for s in c["path"]])
                if key in st.session_state.verified:
                    n = st.session_state.verified[key]
                    (st.success if n == 1 else st.error)(f"{n} matching path(s) in the live graph")
    with tabs[1]:
        if resp["citations"]:
            st.graphviz_chart(subgraph_dot(resp))
        else:
            st.info("No citation paths.")
    with tabs[2]:
        votes = pd.Series(resp["votes"]).value_counts().to_dict() if resp["votes"] else {}
        st.dataframe(pd.DataFrame([{"rank": c["rank"], "id": c["id"], "votes": votes.get(c["id"], ""),
                                    "evidence card": c["card"]} for c in resp["candidates"]]),
                     hide_index=True, width="stretch")
    with tabs[3]:
        g = svc.graph_context(resp["answer_id"]) if resp["answer_id"] else None
        if g is None:
            st.info("Neo4j unavailable or node not found.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("PageRank", f"{g['pagerank']:.2f}", help="gds.pageRank over the corpus_kb graph")
            c2.metric("PageRank percentile", f"{g['pct']:.0%}", help=f"among all {g['label']} nodes")
            c3.metric("Louvain community size", g["size"])
            st.caption(f"degree {g['degree']:.0f} · betweenness {g['betweenness']:.0f} · community "
                       f"{g['community']}")
            st.markdown("Highest-PageRank members of the same community:")
            st.markdown("\n".join(f"- {p}" for p in g["peers"]))
    with tabs[4]:
        for c in resp["calls"]:
            with st.expander(f"{c['step']} · {c['latency_s']:.1f} s · {c.get('completion_tokens')} tokens"):
                st.code(c["prompt"], language=None)
                st.markdown(f"**Output:** `{c['output']}`")


# --- chat ---------------------------------------------------------------------

st.title("CTI attribution with a knowledge graph")
st.caption("Paste a threat-report passage (or a full CTIConnect-style question). The answer is one MITRE "
           "ATT&CK technique or CWE, chosen by Llama-3-8B from snapshot candidates, with the graph paths "
           "behind it.")

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render(m, i)

prompt = st.chat_input("Paste a passage from a threat report…")
gold, report_id = None, None
if ask_example:
    e = ex[pick]
    prompt, gold, report_id = e["question"], e["gold"], e["report_id"]
    tax = Taxonomy.ATTACK if e["task"] == "ata" else Taxonomy.CWE
elif prompt:
    tax = {"MITRE ATT&CK (ATA)": Taxonomy.ATTACK, "CWE (VCA)": Taxonomy.CWE}.get(tax_label) or guess_taxonomy(prompt)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    passage = quoted_passage(prompt)
    req = AttributionRequest(passage=passage, taxonomy=tax, question=prompt if passage != prompt else None,
                             report_id=report_id, method=METHODS[method_label], explain=explain)
    with st.chat_message("assistant"):
        with st.spinner(f"{method_label} · {'ATT&CK' if tax == Taxonomy.ATTACK else 'CWE'} · {svc.model}…"):
            resp: AttributionResponse = svc.attribute(req)
        msg = {"role": "assistant", "response": resp.model_dump(mode="json"), "method": req.method, "gold": gold}
        st.session_state.messages.append(msg)
        render(msg, len(st.session_state.messages) - 1)
