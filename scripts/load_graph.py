#!/usr/bin/env python3
"""Initialize Neo4j from schema.cypher and load the snapshot graph + report layer.

Neo4j 5 runs locally in Docker (credentials in .env):

    docker run -d --name cti-neo4j -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \\
        -e NEO4J_AUTH=neo4j/<password> -e NEO4J_PLUGINS='["graph-data-science","apoc"]' \\
        -v cti-neo4j-data:/data neo4j:5.26
    uv run python scripts/load_graph.py --wipe

Embeddings come from the on-disk cache (no API calls once the indexes are built).
"""

from __future__ import annotations

import argparse
import json
import time

from ctiattr.config import EmbedConfig
from ctiattr.graph_db import GraphDB
from ctiattr.retrieval import Embedder
from ctiattr.snapshot import Snapshot

# Distinct links; CTIFoundry's 1,214 / 272 count three duplicated CAPEC entries twice.
EXPECTED_RELS = {"SUBTECHNIQUE_OF": 518, "IN_TACTIC": 1374, "REPLACED_BY": 103, "EXPLOITS_WEAKNESS": 1212,
                 "MAPS_TO_TECHNIQUE": 271, "HAS_WEAKNESS": 3286, "HAS_CHUNK": 650, "GROUNDS": 3512}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", action="store_true", help="delete all nodes first")
    args = ap.parse_args()
    db = GraphDB()
    t0 = time.time()
    if args.wipe:
        db.wipe()
    print(f"schema.cypher: {db.apply_schema()} statements applied")
    counts = db.load(Snapshot(), Embedder(EmbedConfig.from_env()))
    print(json.dumps(counts, indent=1))
    bad = {k: (counts["relationships"].get(k), v) for k, v in EXPECTED_RELS.items()
           if counts["relationships"].get(k) != v}
    print("edge counts match the Python snapshot" if not bad else f"MISMATCH (got, expected): {bad}")
    print(f"loaded in {time.time() - t0:.0f}s")
    db.close()


if __name__ == "__main__":
    main()
