"""
Fourier control figure of one galaxy: image rotated with the bar on x and the
two bars (isophotes, Fourier), the bar-only image, and A2/A0(r), φ2(r).

Usage:
    python tools/plot_fourier.py --name 39627835815496281 --survey desi --a-ref 14.05
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from matplotlib.patches import Ellipse as EllipsePatch
from scipy.ndimage import rotate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checks import bar_full as bar, fourier_full as fourier
from src import isophotes
from src.cutout import pixscale
A_MIN = 2.0  # arcsec, as in src/measure.py


def centred_rotation(img, x0, y0, angle):
    c = (np.array(img.shape) - 1) / 2
    shifted = np.roll(img, (int(round(c[0] - y0)), int(round(c[1] - x0))), axis=(0, 1))
    return rotate(shifted, angle, reshape=False, order=1), c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--survey", choices=["desi", "sdss"], default="desi")
    p.add_argument("--dir", type=Path, default=Path("outputs/cutouts"))
    p.add_argument("--a-ref", type=float, help="Reference bar semi-length [arcsec]")
    p.add_argument("--zoom", type=float, default=25.0)
    args = p.parse_args()

    ps = pixscale(args.survey)
    stem = args.dir / f"{args.name}_{args.survey}"
    img = fits.getdata(f"{stem}_r.fits").astype(float)
    sky, _, mask, _ = isophotes.sky_and_mask(img)
    img -= sky
    t = Table.read(f"{stem}_iso.ecsv")
    x0, y0 = t["x0"][0], t["y0"][0]

    a, eps, pa = bar.profiles(t, ps)
    iso = bar.bar_from_profile(a, eps, pa, A_MIN)
    r, a0, am, bm = fourier.polar_modes(img, mask, x0, y0, rmax=2 * args.zoom / ps)
    # r, a_phase and W are in px; the profile panels are in arcsec
    fb = fourier.bar_length(r, a0, am, bm, r_min=A_MIN / ps)
    fw = fourier.bar_width(img.shape, x0, y0, r, am, bm, fb["a_phase"], fb["phi2_bar"])

    rot, c = centred_rotation(img, x0, y0, fb["phi2_bar"])
    bo, _ = centred_rotation(fourier.bar_only_image(img.shape, x0, y0, r, am, bm, fb["a_phase"]),
                             x0, y0, fb["phi2_bar"])
    h = args.zoom / ps

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 3)
    ax_rot, ax_bo = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    ax_a2, ax_ph = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])

    norm = ImageNormalize(rot, interval=PercentileInterval(99.9), stretch=AsinhStretch(0.02))
    ax_rot.imshow(rot, origin="lower", cmap="gray_r", norm=norm)
    ax_rot.add_patch(EllipsePatch((c[1], c[0]), 2 * iso["a_eps"] / ps, 2 * iso["b_bar"] / ps,
                                  angle=iso["pa_bar"] - fb["phi2_bar"], fill=False, lw=1.5,
                                  color="tab:blue", label="isophotes (ε max)"))
    ax_rot.add_patch(EllipsePatch((c[1], c[0]), 2 * fb["a_phase"], fw, angle=0,
                                  fill=False, lw=1.5, color="tab:orange", label="Fourier (φ2, m≥2)"))
    if args.a_ref:
        for s in (-1, 1):
            ax_rot.axvline(c[1] + s * args.a_ref / ps, color="k", ls="--", lw=0.8)
    ax_rot.legend(loc="lower left", fontsize=8)
    ax_rot.set_title(f"TARGETID {args.name} ({args.survey.upper()}), bar on x")

    lim = np.nanpercentile(np.abs(bo), 99.5)
    ax_bo.imshow(bo, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax_bo.add_patch(EllipsePatch((c[1], c[0]), 2 * fb["a_phase"], fw, angle=0,
                                 fill=False, lw=1.5, color="tab:orange"))
    ax_bo.set_title("bar-only image: modes m = 2, 4, 6 (no bulge)")
    for ax in (ax_rot, ax_bo):
        ax.set_xlim(c[1] - h, c[1] + h), ax.set_ylim(c[0] - h, c[0] + h)
        ax.set_xticks([]), ax.set_yticks([])

    ra = r * ps
    ax_a2.plot(ra, fb["a2"], color="tab:orange")
    ax_ph.plot(ra, fb["phi2"], ".", ms=3, color="tab:orange")
    ax_ph.axhspan(fb["phi2_bar"] - 10, fb["phi2_bar"] + 10, color="tab:orange", alpha=0.15)
    ax_a2.set_ylabel("A2 / A0"), ax_ph.set_ylabel("φ2 [deg]"), ax_ph.set_xlabel("r [arcsec]")
    for ax in (ax_a2, ax_ph):
        ax.set_xlim(0, 2 * args.zoom)
        ax.axvspan(0, A_MIN, color="0.9")
        ax.axvline(fb["a_phase"] * ps, color="tab:orange", lw=1)
        ax.axvline(iso["a_eps"], color="tab:blue", lw=1)
        if args.a_ref:
            ax.axvline(args.a_ref, color="k", ls="--", lw=0.8)
    fig.tight_layout()
    out = Path(f"{stem}_fourier.png")
    fig.savefig(out, dpi=120)
    print(out, f"a_phase = {fb['a_phase'] * ps:.2f}″, W = {fw * ps:.2f}″")


if __name__ == "__main__":
    main()
