"""
All galaxies measured with the Alonso et al. (2024) method (training
catalogue in modelo/), ready for measure_bars.py, whether or not they are
in our DESI DR1 parent sample.

Galaxies of the parent sample keep their DESI TARGETID, position, z and
Legacy SHAPE_R. The others (no DR1 spectrum within 1″) take the catalogue
position and SDSS z, SHAPE_R_SIZE × petroR90_r as the size of the cutout
(the median SHAPE_R / petroR90_r of the galaxies that have both), and a negative
TARGETID = -id_desi of the catalogue, so they are never confused with DESI ids.

Usage:
    python scripts/build_reference_sample.py --output outputs/sample/reference_all.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import write_provenance

REF = {"a_bar_arcsec": "a_ref", "b_bar_arcsec": "b_ref", "id_desi": "id_ref",
       "kpc_per_arcsec": "kpc_per_arcsec_ref"}
SHAPE_R_SIZE = 0.66


def main():
    p = argparse.ArgumentParser(description="Reference sample with DESI ids where available")
    p.add_argument("--sample", type=Path, default=Path("outputs/sample/barred_sample.parquet"))
    p.add_argument("--reference", type=Path, default=Path("modelo/entrenamiento_parametros"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    ref = pd.read_csv(args.reference, usecols=["dr8_id", "ra", "dec", "z_1", "petroR90_r", *REF])
    ref = ref.rename(columns=REF)
    desi = pd.read_parquet(args.sample, columns=["TARGETID", "RA", "DEC", "Z", "SHAPE_R", "dr8_id"])
    # nullable integer: a left merge would turn TARGETID into float64, which cannot hold 17 digits
    desi["TARGETID"] = desi["TARGETID"].astype("Int64")
    out = ref.merge(desi, on="dr8_id", how="left", validate="one_to_one")
    out = out.astype({c: "float64" for c in ["RA", "DEC", "Z", "SHAPE_R"]})

    out["in_desi"] = out["TARGETID"].notna()
    new = ~out["in_desi"]
    out.loc[new, "TARGETID"] = -out.loc[new, "id_ref"]
    out.loc[new, "RA"], out.loc[new, "DEC"] = out.loc[new, "ra"], out.loc[new, "dec"]
    out.loc[new, "Z"] = out.loc[new, "z_1"]
    out.loc[new, "SHAPE_R"] = SHAPE_R_SIZE * out.loc[new, "petroR90_r"]
    out["TARGETID"] = out["TARGETID"].astype("int64")
    out = out.drop(columns=["ra", "dec", "z_1", "petroR90_r"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    write_provenance(args.output, "build_reference_sample.py",
                     {"sample": args.sample, "reference": args.reference}, {},
                     {"reference": len(ref), "in_desi": int(out["in_desi"].sum()), "new": int(new.sum())})
    print(len(out), int(out["in_desi"].sum()), int(new.sum()))


if __name__ == "__main__":
    main()
