"""
2D decomposition (bulge + disk + bar + ring) of galaxies already measured by
measure_bars.py, whose isophotes give the starting values.

Usage:
    python scripts/decompose_bars.py --sample outputs/sample/common_valeria.parquet \\
        --bars outputs/bars/bars_common_desi.parquet --output outputs/bars/decomp.parquet \\
        --targetid 39627884146461249
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checks import decompose
from src import isophotes
from src.cutout import cutout_paths, pixscale
from src.utils import write_provenance


def _one(row, b, cut_dir, survey):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = pixscale(survey)
            name = str(row.TARGETID)
            img = fits.getdata(cutout_paths(cut_dir, name, survey)["fits"]).astype(float)
            psf, ivar = decompose.download_psf_ivar(row.RA, row.DEC, img.shape[0], cut_dir, name, survey)
            sky, _, mask, galaxy = isophotes.sky_and_mask(img)
            img = img - sky
            x0, y0 = isophotes.galaxy_centre(img, galaxy)
            iso = dict(a_eps=b.a_eps / ps, eps_bar=b.eps_bar, pa_bar=b.pa_bar,
                       eps_disk=b.eps_disk, pa_disk=b.pa_disk, h0=row.SHAPE_R / 1.68 / ps)
            p, chi2, ok, model = decompose.fit(img, ivar, mask, psf,
                                               decompose.model_config(x0, y0, iso, img, ps))
            fits.writeto(cut_dir / f"{name}_{survey}_model.fits", model.astype(np.float32), overwrite=True)
            out = {"TARGETID": row.TARGETID, "chi2r": chi2, "converged": ok}
            out.update({k.replace(":", "_"): v for k, v in p.items()})
            out["bar_a"] = p["bar:a_bar"] * ps
            out["bar_b"] = p["bar:a_bar"] * (1 - p["bar:ell"]) * ps
            out["ring_R"] = p["ring:R_ring"] * ps
            return out
    except Exception as err:
        return str(row.TARGETID), repr(err)


def main():
    p = argparse.ArgumentParser(description="2D decomposition of measured bars")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--bars", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cutouts", type=Path, default=Path("outputs/cutouts"))
    p.add_argument("--survey", choices=["desi", "sdss"], default="desi")
    p.add_argument("--targetid", type=int, nargs="*")
    p.add_argument("--n-jobs", type=int, default=1)
    args = p.parse_args()

    bars = pd.read_parquet(args.bars).set_index("TARGETID")
    sample = pd.read_parquet(args.sample)
    sample = sample[sample["TARGETID"].isin(args.targetid or bars.index)]
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(_one)(row, bars.loc[row.TARGETID], args.cutouts, args.survey) for row in sample.itertuples())
    rows = [r for r in res if isinstance(r, dict)]
    failed = dict(r for r in res if isinstance(r, tuple))
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    write_provenance(args.output, "decompose_bars.py", {"sample": args.sample, "bars": args.bars},
                     {"survey": args.survey, "targetid": args.targetid},
                     {"input": len(sample), "fitted": len(rows), "failed": failed})
    print(f"fitted {len(rows)} / {len(sample)}, failed {len(failed)}")


if __name__ == "__main__":
    main()
