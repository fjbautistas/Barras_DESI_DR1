"""
Bar control figure of one galaxy:
image with the bar ellipse, the image rotated so that the bar lies on x,
and the ε(a), PA(a), μ(a) profiles with the bar limits.

Usage:
    python tools/plot_profiles.py --name 39627835815496281 --survey desi --a-ref 14.05
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from matplotlib.patches import Ellipse as EllipsePatch, Rectangle
from scipy.ndimage import rotate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checks import bar_full as bar
from src.cutout import pixscale
A_MIN = 2.0  # arcsec, as in src/measure.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--survey", choices=["desi", "sdss"], default="desi")
    p.add_argument("--dir", type=Path, default=Path("outputs/cutouts"))
    p.add_argument("--a-ref", type=float, help="Reference bar semi-length [arcsec]")
    p.add_argument("--zoom", type=float, default=50.0, help="Half side of the image panels [arcsec]")
    args = p.parse_args()

    PIXSCALE = pixscale(args.survey)
    stem = args.dir / f"{args.name}_{args.survey}"
    img = fits.getdata(f"{stem}_r.fits")
    t = Table.read(f"{stem}_iso.ecsv")
    a, eps, pa = bar.profiles(t, PIXSCALE)
    eps_err = np.array(t["ellipticity_err"])[np.isin(t["stop_code"], bar.GOOD_STOP)]
    mu = 22.5 - 2.5 * np.log10(np.clip(np.array(t["intens"]), 1e-6, None) / PIXSCALE**2)
    mu = mu[np.isin(t["stop_code"], bar.GOOD_STOP)]
    b = bar.bar_from_profile(a, eps, pa, A_MIN)
    eps_d, pa_d, _ = bar.disk_orientation(a, eps, pa)
    x0, y0 = t["x0"][0], t["y0"][0]
    h = args.zoom / PIXSCALE
    norm = ImageNormalize(img, interval=PercentileInterval(99.9), stretch=AsinhStretch(0.02))

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(3, 3)
    ax_img, ax_rot = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    ax_eps, ax_pa, ax_mu = (fig.add_subplot(gs[k, 2]) for k in range(3))

    ax_img.imshow(img, origin="lower", cmap="gray_r", norm=norm)
    to_px = 1 / PIXSCALE
    ax_img.add_patch(EllipsePatch((x0, y0), 2 * b["a_eps"] * to_px, 2 * b["b_bar"] * to_px,
                                  angle=b["pa_bar"], fill=False, lw=1.5, color="tab:blue"))
    ax_img.add_patch(EllipsePatch((x0, y0), 2 * a.max() * to_px, 2 * a.max() * (1 - eps_d) * to_px,
                                  angle=pa_d, fill=False, lw=1.0, ls="--", color="tab:purple"))
    ax_img.set_xlim(x0 - h, x0 + h), ax_img.set_ylim(y0 - h, y0 + h)
    ax_img.set_title(f"TARGETID {args.name}  ({args.survey.upper()})")

    # rotation about the galaxy centre: shift it to the array centre first.
    # ndimage.rotate turns counter-clockwise in (row, col) order, which is
    # clockwise when shown with origin="lower": +pa_bar brings the bar to +x.
    c = (np.array(img.shape) - 1) / 2
    shifted = np.roll(img, (int(round(c[0] - y0)), int(round(c[1] - x0))), axis=(0, 1))
    rot = rotate(shifted, b["pa_bar"], reshape=False, order=1)
    ax_rot.imshow(rot, origin="lower", cmap="gray_r", norm=norm)
    ax_rot.add_patch(Rectangle((c[1] - b["a_eps"] * to_px, c[0] - b["b_bar"] * to_px),
                               2 * b["a_eps"] * to_px, 2 * b["b_bar"] * to_px,
                               fill=False, lw=1.5, color="tab:blue"))
    for s in (-1, 1):
        ax_rot.axvline(c[1] + s * b["a_up"] * to_px, color="tab:blue", ls=":", lw=1)
        if args.a_ref:
            ax_rot.axvline(c[1] + s * args.a_ref * to_px, color="k", ls="--", lw=0.8)
    ax_rot.set_xlim(c[1] - h, c[1] + h), ax_rot.set_ylim(c[0] - h, c[0] + h)
    ax_rot.set_title("bar on x: 2a × 2b (solid), a_up (dotted)")
    for ax in (ax_img, ax_rot):
        ax.set_xticks([]), ax.set_yticks([])

    ax_eps.errorbar(a, eps, eps_err, fmt="o", ms=2, color="tab:blue")
    ax_pa.plot(a, pa, "o", ms=2, color="tab:blue")
    ax_mu.plot(a, mu, "o", ms=2, color="tab:blue")
    ax_eps.axhline(eps_d, color="tab:purple", ls="--", lw=0.8)
    ax_pa.axhline(pa_d, color="tab:purple", ls="--", lw=0.8)
    ax_eps.set_ylabel("ε"), ax_pa.set_ylabel("PA [deg]"), ax_mu.set_ylabel("μ_r [mag/″²]")
    ax_mu.invert_yaxis(), ax_mu.set_xlabel("a [arcsec]")
    for ax in (ax_eps, ax_pa, ax_mu):
        ax.set_xlim(0, args.zoom)
        ax.axvspan(0, A_MIN, color="0.9")
        ax.axvline(b["a_eps"], color="tab:blue", lw=1)
        ax.axvline(b["a_up"], color="tab:blue", ls=":", lw=1)
        if args.a_ref:
            ax.axvline(args.a_ref, color="k", ls="--", lw=0.8)
    fig.tight_layout()
    out = Path(f"{stem}_bar.png")
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
