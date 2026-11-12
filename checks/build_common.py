"""
Galaxies of the parent sample also measured by Alonso et al. (2024) method
(training catalogue in modelo/), matched by dr8_id, with their bar
semi-axes in arcsec (projected, measured on SDSS JPG at 0.302″/px).

Usage:
    python scripts/build_common.py --output outputs/sample/common_valeria.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import write_provenance

REF_COLUMNS = {"a_bar_arcsec": "a_ref", "b_bar_arcsec": "b_ref", "id_desi": "id_ref"}


def main():
    p = argparse.ArgumentParser(description="Common sample with the reference measurements")
    p.add_argument("--sample", type=Path, default=Path("outputs/sample/barred_sample.parquet"))
    p.add_argument("--reference", type=Path, default=Path("modelo/entrenamiento_parametros"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    sample = pd.read_parquet(args.sample)
    ref = pd.read_csv(args.reference, usecols=["dr8_id", *REF_COLUMNS]).rename(columns=REF_COLUMNS)
    common = sample.merge(ref, on="dr8_id", validate="one_to_one")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.to_parquet(args.output, index=False)
    write_provenance(args.output, "build_common.py",
                     {"sample": args.sample, "reference": args.reference}, {},
                     {"sample": len(sample), "reference": len(ref), "common": len(common)})
    print(len(common))


if __name__ == "__main__":
    main()
