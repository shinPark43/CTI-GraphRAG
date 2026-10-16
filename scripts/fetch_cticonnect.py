#!/usr/bin/env python3
"""Download the CTIConnect files this project uses, pinned to one commit.

CTIConnect (Cheng et al., KDD '26) — data CC-BY-4.0, code MIT.
https://github.com/peng-gao-lab/CTIConnect

    uv run python scripts/fetch_cticonnect.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

COMMIT = "554797d69a51147f1f98fad7198cb2d2b183d0e9"
BASE = f"https://raw.githubusercontent.com/peng-gao-lab/CTIConnect/{COMMIT}"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "cticonnect"

FILES = [
    "data/manifest.json",
    "data/entity_attribution/ata.jsonl",
    "data/entity_attribution/vca.jsonl",
    "corpus_kb/MANIFEST.json",
    "corpus_kb/cwe.jsonl",
    "corpus_kb/mitre.jsonl",
    "corpus_kb/capec.jsonl",
    "corpus_kb/cve.jsonl",
    "corpus_reports/preprocessed_reports.jsonl",
    "LICENSE",
    "LICENSE-DATA",
]


def fetch(rel: str) -> Path:
    dest = OUT / rel
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{BASE}/{rel}", timeout=120) as r:
        dest.write_bytes(r.read())
    print(f"  fetched {rel}")
    return dest


def verify_kb() -> None:
    manifest = json.loads((OUT / "corpus_kb/MANIFEST.json").read_text())
    for name, meta in manifest["files"].items():
        digest = hashlib.sha256((OUT / "corpus_kb" / name).read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            sys.exit(f"sha256 mismatch for corpus_kb/{name}")
    print("  corpus_kb checksums OK")


def main() -> None:
    print(f"CTIConnect @ {COMMIT[:7]} -> {OUT}")
    for rel in FILES:
        fetch(rel)
    verify_kb()
    (OUT / "COMMIT").write_text(COMMIT + "\n")


if __name__ == "__main__":
    main()
