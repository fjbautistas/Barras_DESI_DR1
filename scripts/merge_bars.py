"""
Join the blocks written by measure_bars.py --n-chunks into one table.

Usage:
    python scripts/merge_bars.py --output outputs/bars/bars.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import write_provenance


def main():
    p = argparse.ArgumentParser(description="Join measure_bars.py blocks")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    chunks = sorted(args.output.parent.glob(f"{args.output.stem}_chunk*.parquet"))
    bars = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
    if not bars["TARGETID"].is_unique:
        raise SystemExit("repeated TARGETID: some block was measured twice")
    failed = {}
    for c in chunks:
        failed.update(json.loads(c.with_suffix(".json").read_text())["counts"]["failed"])
    bars.to_parquet(args.output, index=False)
    write_provenance(args.output, "merge_bars.py", {c.name: c for c in chunks}, {},
                     {"chunks": len(chunks), "measured": len(bars), "failed": failed})
    print(f"{len(chunks)} blocks, {len(bars)} bars, {len(failed)} failed")


if __name__ == "__main__":
    main()
