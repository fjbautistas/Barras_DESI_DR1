"""
The same method on the FITS (linear flux, band r) and on the colour JPG
(green channel = r, 8 bits, non-linear stretch) of one galaxy.

Usage:
    python checks/jpg_vs_fits.py --targetid 39627835815496281
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bar
from src.cutout import cutout_paths, pixscale
from src.measure import measure_image
from tools.steps import _show, boxy

CUT = ROOT / "outputs/cutouts"
SCRATCH = ROOT / "outputs/checks"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targetid", default="39627835815496281")
    p.add_argument("--z", type=float, default=0.039)
    args = p.parse_args()
    ps = pixscale("desi")
    paths = cutout_paths(CUT, args.targetid, "desi")
    SCRATCH.mkdir(parents=True, exist_ok=True)

    imgs = {"FITS r": fits.getdata(paths["fits"]).astype(float),
            "JPG verde": np.flipud(np.asarray(Image.open(paths["jpg"]))[:, :, 1]).astype(float)}
    colors = {"FITS r": "tab:blue", "JPG verde": "tab:purple"}
    res, prof = {}, {}
    for name, img in imgs.items():
        iso = SCRATCH / f"{args.targetid}_{name.split()[0].lower()}_iso.ecsv"
        res[name] = measure_image(img, ps, args.z, iso_path=iso)
        prof[name] = bar.profiles(Table.read(iso), ps)
        r = res[name]
        print(f"{name:10s} a = {r['a_bar']:.2f}″  eps = {r['eps_bar']:.2f}  PA = {r['pa_bar']:.1f}°  "
              f"W = {r['W_bar']:.2f}″  2b_iso = {2 * r['b_iso']:.2f}″  flags eps/end/pa = "
              f"{r['flag_eps']}/{r['flag_end']}/{r['flag_pa']}")

    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.3])
    h = 1.8 * res["FITS r"]["a_bar"] / ps
    for k, name in enumerate(imgs):
        ax = fig.add_subplot(gs[:, k])
        x0, y0 = res[name]["x0"], res[name]["y0"]
        _show(ax, imgs[name], x0, y0, h)
        for other, r in res.items():
            ax.plot(*boxy(r["a_bar"] / ps, r["W_bar"] / 2 / ps, r["pa_bar"] + 90, x0, y0),
                    color=colors[other], lw=1.5, ls="-" if other == name else "--")
        ax.set_title(f"{name}: a = {res[name]['a_bar']:.1f}″, W = {res[name]['W_bar']:.1f}″", fontsize=9)
    ax_e, ax_p = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    for name, (a, eps, pa) in prof.items():
        ax_e.plot(a, eps, "o", ms=2.5, color=colors[name], label=name)
        ax_p.plot(a, pa, "o", ms=2.5, color=colors[name])
        for ax in (ax_e, ax_p):
            ax.axvline(res[name]["a_bar"], color=colors[name], lw=1)
    ax_e.set_ylabel("ε"), ax_p.set_ylabel("PA imagen [°]"), ax_p.set_xlabel("a [″]")
    ax_e.set_xlim(0, 3 * res["FITS r"]["a_bar"]), ax_p.set_xlim(ax_e.get_xlim())
    ax_e.legend(fontsize=8)
    fig.suptitle("Continua: barra medida en esa imagen; a trazos: la de la otra", fontsize=9)
    fig.tight_layout()
    out = ROOT / "outputs/gallery" / f"{args.targetid}_jpg_vs_fits.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
