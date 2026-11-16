"""
Input of measure_bars.py from any list of galaxies (CSV or parquet).

Only RA and DEC are required. A missing TARGETID becomes the row number,
a missing Z is left as NaN (sizes in arcsec only), and a missing SHAPE_R
is taken from the nearest Legacy DR10 Tractor source within 1″. Positions
with no Tractor source cannot be sized and are dropped (listed in the .json).
Run it on a machine with internet, before the cluster.

Usage:
    python scripts/build_input.py --list galaxies.csv --output outputs/sample/input.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cutout import tractor_shape
from src.utils import write_provenance


def main():
    p = argparse.ArgumentParser(description="Input sample for measure_bars.py")
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ra", default="RA")
    p.add_argument("--dec", default="DEC")
    args = p.parse_args()

    raw = pd.read_parquet(args.list) if args.list.suffix == ".parquet" else pd.read_csv(args.list)
    s = raw.rename(columns={args.ra: "RA", args.dec: "DEC"})
    s["TARGETID"] = s["TARGETID"] if "TARGETID" in s else np.arange(len(s))
    s["Z"] = s["Z"] if "Z" in s else np.nan
    s["SHAPE_R"] = s["SHAPE_R"] if "SHAPE_R" in s else np.nan
    miss = s["SHAPE_R"].isna().to_numpy()
    if miss.any():
        tr = tractor_shape(s.loc[miss, "RA"], s.loc[miss, "DEC"])
        s.loc[miss, "SHAPE_R"] = tr["SHAPE_R"].to_numpy()
        s["TRACTOR_TYPE"] = None
        s.loc[miss, "TRACTOR_TYPE"] = tr["TYPE"].to_numpy()
    lost = s.loc[s["SHAPE_R"].isna(), "TARGETID"].tolist()
    s = s[s["SHAPE_R"].notna()]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(args.output, index=False)
    write_provenance(args.output, "build_input.py", {"list": args.list}, {"tractor_radius_arcsec": 1.0},
                     {"input": len(raw), "shape_from_tractor": int(miss.sum()) - len(lost),
                      "written": len(s), "no_tractor_source": lost})
    print(f"{len(s)} / {len(raw)} written; SHAPE_R from Tractor for {int(miss.sum()) - len(lost)}, no source for {len(lost)}")


if __name__ == "__main__":
    main()
