"""
Step 1: parent sample of barred galaxies in DESI DR1.

GZ DESI galaxies with p_bar > pbar_min, matched to DESI DR1 within tol_arcsec,
at z < z_max (the bar must be resolved), with their Legacy shape.

Usage:
    python scripts/build_sample.py --output outputs/sample/barred_sample.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sample import add_shape, load_gz_bars, match_desi
from src.utils import write_provenance

PHD = Path("/Users/frank/Documents/PhD/PhD_tesis")
GZ_DESI = PHD / "semestre-2026-II/Tesis_III/Comparison/outputs/gz_desi_subset.parquet"
DESI = PHD / "Semestre-2025-II/Tesis_I/Galaxy_Pairs/Galaxy_pairs/data"


def main():
    p = argparse.ArgumentParser(description="Parent sample of barred galaxies")
    p.add_argument("--gz", type=Path, default=GZ_DESI)
    p.add_argument("--base", type=Path, default=DESI / "DESI_galaxies_base.parquet")
    p.add_argument("--aux", type=Path, default=DESI / "DESI_galaxies_aux.parquet")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pbar-min", type=float, default=0.5)
    p.add_argument("--z-max", type=float, default=0.1)
    p.add_argument("--tol-arcsec", type=float, default=1.0)
    args = p.parse_args()

    gz = load_gz_bars(args.gz, args.pbar_min)
    base = pd.read_parquet(args.base)
    matched = match_desi(gz, base, args.tol_arcsec)
    sample = matched[matched["Z"] < args.z_max]
    sample = add_shape(sample, args.aux)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output, index=False)
    counts = {"gz_pbar": len(gz), "matched_dr1": len(matched), "sample": len(sample)}
    write_provenance(args.output, "build_sample.py",
                     {"gz": args.gz, "base": args.base, "aux": args.aux},
                     {"pbar_min": args.pbar_min, "z_max": args.z_max,
                      "tol_arcsec": args.tol_arcsec}, counts)
    print(counts)


if __name__ == "__main__":
    main()
