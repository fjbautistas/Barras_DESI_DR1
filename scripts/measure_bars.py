"""
Bar length and width of every galaxy of a sample.

Usage:
    python scripts/measure_bars.py --sample outputs/sample/barred_sample.parquet \\
        --output outputs/bars/bars.parquet --n-jobs 8

On a cluster, download first on a node with internet, then measure in blocks
(one job per block) and join them with scripts/merge_bars.py:
    python scripts/measure_bars.py --sample ... --output ... --download-only --n-jobs 4
    python scripts/measure_bars.py --sample ... --output outputs/bars/bars.parquet \\
        --n-chunks 100 --chunk $SLURM_ARRAY_TASK_ID --n-jobs $SLURM_CPUS_PER_TASK
Block k is written as bars_chunk{k:03d}.parquet next to --output.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import cutout
from src.measure import A_MIN, measure_galaxy
from src.utils import write_provenance


def _run(row, args):
    """One galaxy; a failure is returned as (TARGETID, error) so the others go on."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.download_only:
                size = int(cutout.cutout_size(row.SHAPE_R, args.survey))
                cutout.download(row.RA, row.DEC, size, args.cutouts, str(row.TARGETID), args.survey)
                return {"TARGETID": row.TARGETID}
            return measure_galaxy(row, args.cutouts, args.survey, args.a_min)
    except Exception as err:
        return str(row.TARGETID), repr(err)


def main():
    p = argparse.ArgumentParser(description="Bar measurements")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cutouts", type=Path, default=Path("outputs/cutouts"))
    p.add_argument("--survey", choices=["desi", "sdss"], default="desi")
    p.add_argument("--targetid", type=int, nargs="*", help="Only these galaxies")
    p.add_argument("--a-min", type=float, default=A_MIN)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--n-chunks", type=int, default=1)
    p.add_argument("--chunk", type=int, default=0, help="Block to measure, 0 … n-chunks − 1")
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()

    sample = pd.read_parquet(args.sample)
    if args.targetid:
        sample = sample[sample["TARGETID"].isin(args.targetid)]
    if args.n_chunks > 1:
        sample = sample.iloc[np.array_split(np.arange(len(sample)), args.n_chunks)[args.chunk]]
        args.output = args.output.with_name(f"{args.output.stem}_chunk{args.chunk:03d}.parquet")

    results = Parallel(n_jobs=args.n_jobs)(delayed(_run)(row, args) for row in sample.itertuples())
    rows = [r for r in results if isinstance(r, dict)]
    failed = dict(r for r in results if isinstance(r, tuple))
    print(f"{'downloaded' if args.download_only else 'measured'} {len(rows)} / {len(sample)}, failed {len(failed)}")
    if args.download_only:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    write_provenance(args.output, "measure_bars.py", {"sample": args.sample},
                     {"survey": args.survey, "a_min_arcsec": args.a_min, "targetid": args.targetid,
                      "chunk": args.chunk, "n_chunks": args.n_chunks},
                     {"input": len(sample), "measured": len(rows), "failed": failed})


if __name__ == "__main__":
    main()
