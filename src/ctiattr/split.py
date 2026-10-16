"""Dev/test split by source report.

ATA and VCA questions are split jointly by report, so no report contributes
questions to both sides (a report has up to 4 questions across the two tasks).

    uv run python -m ctiattr.split
"""

from __future__ import annotations

import json
import random
from collections import Counter

from ctiattr.config import SPLITS
from ctiattr.data import load_all

SEED = 13
DEV_FRACTION = 0.2
SPLIT_FILE = SPLITS / "split_v1.json"


def make_split() -> dict:
    qas = load_all()
    reports = sorted({qa.report_id for qa in qas}, key=lambda r: int(r.split("-")[1]))
    rng = random.Random(SEED)
    rng.shuffle(reports)
    dev_reports = set(reports[: round(len(reports) * DEV_FRACTION)])
    split = {
        "seed": SEED,
        "dev_fraction": DEV_FRACTION,
        "dev_reports": sorted(dev_reports, key=lambda r: int(r.split("-")[1])),
        "dev": [qa.id for qa in qas if qa.report_id in dev_reports],
        "test": [qa.id for qa in qas if qa.report_id not in dev_reports],
    }
    return split


def load_split() -> dict:
    split = json.loads(SPLIT_FILE.read_text())
    if MINI_FILE.exists():
        split["mini20"] = json.loads(MINI_FILE.read_text())["ids"]
    return split


MINI_FILE = SPLITS / "mini20.json"
MINI_PER_TASK = 10


def make_mini() -> dict:
    """20-question quick-look subset of dev: 10 ATA + 10 VCA, fixed seed."""
    dev = set(load_split()["dev"])
    rng = random.Random(SEED)
    ids = []
    for task in ("ata", "vca"):
        pool = sorted(i for i in dev if i.startswith(task))
        ids += sorted(rng.sample(pool, MINI_PER_TASK))
    return {"seed": SEED, "source": "dev", "ids": ids}


def main() -> None:
    if not MINI_FILE.exists() and SPLIT_FILE.exists():
        MINI_FILE.write_text(json.dumps(make_mini(), indent=1) + "\n")
        print(f"wrote {MINI_FILE}")
        return
    if SPLIT_FILE.exists():
        raise SystemExit(f"{SPLIT_FILE} already exists; the split is frozen")
    split = make_split()
    SPLITS.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=1) + "\n")
    for side in ("dev", "test"):
        c = Counter(i.split("-")[0] for i in split[side])
        print(f"{side}: {len(split[side])} questions {dict(c)}")
    print(f"dev reports: {len(split['dev_reports'])}")


if __name__ == "__main__":
    main()
