#!/usr/bin/env python3
"""Run gds.cypher on the corpus_kb graph and summarize its topology.

Writes degree, component, pagerank, community and betweenness onto the nodes,
then reports: degree by label, connected components, PageRank and betweenness
leaders, Louvain communities (and how well they match ATT&CK tactics), coverage
of the CVE -> CWE <- CAPEC -> technique bridge, and one shortest path.
No questions are used here.

    uv run python scripts/gds_analysis.py     # -> runs/gds_analysis.json
"""

from __future__ import annotations

import json
import time

from ctiattr import gds
from ctiattr.config import RUNS
from ctiattr.graph_db import GraphDB


def main() -> None:
    db = GraphDB()
    t0 = time.time()
    steps = gds.run_all(db)
    secs = time.time() - t0
    out = {
        "seconds": round(secs, 1),
        "graph": steps["project"],
        "louvain": {k: steps["louvain"][k] for k in ("communityCount", "modularity", "ranLevels")},
        "wcc": {"componentCount": steps["wcc"]["componentCount"]},
        "pagerank": {k: steps["pagerank"][k] for k in ("ranIterations", "didConverge")},
        "degree_by_label": gds.degree_by_label(db),
        "components": gds.components(db),
        "pagerank_top": {l: gds.top_nodes(db, "pagerank", l) for l in ("AttackTechnique", "CWE", "Tactic")},
        "betweenness_top": {l: gds.top_nodes(db, "betweenness", l) for l in ("CWE", "CAPEC", "AttackTechnique")},
        "communities": gds.communities(db),
        "community_tactic_alignment": gds.community_purity(db),
        "bridges": gds.bridges(db),
    }
    pair = gds.example_bridge(db)
    out["example_path"] = gds.shortest_path(db, *pair) if pair else None

    g = out["graph"]
    print(f"kb graph: {g['nodeCount']} nodes, {g['relationshipCount'] // 2} edges (undirected); "
          f"all algorithms {secs:.1f}s")
    print(f"WCC: {out['wcc']['componentCount']} components, largest {out['components'][0]['size']}")
    print(f"Louvain: {out['louvain']['communityCount']} communities, modularity {out['louvain']['modularity']:.3f}; "
          f"tactic alignment {out['community_tactic_alignment']['share']:.1%}")
    print("degree by label:")
    for l, d in out["degree_by_label"].items():
        print(f"  {l:16} n={d['nodes']:5}  mean {d['mean']:6.2f}  median {d['median']:4.0f}  max {d['max']:4.0f}  "
              f"isolated {d['isolated']}")
    print("PageRank leaders:")
    for l, rows in out["pagerank_top"].items():
        print(f"  {l:16} " + ", ".join(f"{r['id']} ({r['pagerank']:.1f})" for r in rows[:5]))
    print("betweenness leaders:")
    for l, rows in out["betweenness_top"].items():
        print(f"  {l:16} " + ", ".join(r["id"] for r in rows[:5]))
    print("largest communities:")
    for c in out["communities"][:6]:
        lead = next(iter(c["leaders"].get("AttackTechnique") or c["leaders"].get("CWE") or [""]))
        print(f"  {c['size']:4}  {c['labels']}  lead: {lead[:60]}")
    print(f"bridge: {json.dumps(out['bridges'])}")
    print(f"example path: {out['example_path']}")

    path = RUNS / "gds_analysis.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
