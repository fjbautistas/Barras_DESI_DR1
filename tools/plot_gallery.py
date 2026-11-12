"""
Gallery of measured bars, drawn as a boxy ellipse |x/a|³ + |y/b|³ = 1
(illustrative shape; a, b, PA measured).
Purple: the measured bar (length and PA from the isophotes, width from the
bulge-free Fourier image). Dashed black circle: reference a, if given.

Usage:
    python tools/plot_gallery.py --bars outputs/bars/bars_common_desi.parquet \\
        --reference outputs/sample/common_valeria.parquet --out-dir outputs/gallery
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from matplotlib.patches import Circle
from scipy.ndimage import rotate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cutout import cutout_paths, pixscale


def boxy(a, b, pa_img, x0, y0, c=3.0, n=300):
    """Generalised ellipse |x/a|^c + |y/b|^c = 1 rotated by pa_img [deg, from +x CCW]."""
    t = np.linspace(0, 2 * np.pi, n)
    x = a * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / c)
    y = b * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / c)
    p = np.deg2rad(pa_img)
    return x0 + x * np.cos(p) - y * np.sin(p), y0 + x * np.sin(p) + y * np.cos(p)


def bar_on_x(img, g):
    """Image moved to the galaxy centre and rotated so the bar (isophote PA) lies on +x.

    ndimage.rotate turns counter-clockwise in (row, col), i.e. clockwise as
    shown with origin="lower", so +pa_img brings the bar to the x axis.
    """
    c = (np.array(img.shape) - 1) / 2
    shifted = np.roll(img, (int(round(c[0] - g.y0)), int(round(c[1] - g.x0))), axis=(0, 1))
    return rotate(shifted, g.pa_bar + 90, reshape=False, order=1), c[1], c[0]


def panel(ax, g, cut_dir, survey, on_x=False, cmap="gray_r"):
    ps = pixscale(survey)
    img = fits.getdata(cutout_paths(cut_dir, str(g.TARGETID), survey)["fits"]) - g.sky
    x0, y0, turn = g.x0, g.y0, 0.0
    if on_x:
        img, x0, y0 = bar_on_x(img, g)
        turn = g.pa_bar + 90
    norm = ImageNormalize(img, interval=PercentileInterval(99.8), stretch=AsinhStretch(0.03))
    ax.imshow(img, origin="lower", cmap=cmap, norm=norm)
    ax.plot(*boxy(g.a_bar / ps, g.W_bar / 2 / ps, g.pa_bar + 90 - turn, x0, y0), color="tab:purple", lw=1.6)
    a_ref = getattr(g, "a_ref", np.nan)
    if np.isfinite(a_ref):
        ax.add_patch(Circle((x0, y0), a_ref / ps, fill=False, color="k", ls="--", lw=0.8))
    h = 1.6 * np.nanmax([g.a_bar, a_ref]) / ps
    ax.set_xlim(x0 - h, x0 + h), ax.set_ylim(y0 - h, y0 + h)
    ax.set_xticks([]), ax.set_yticks([])
    ref = f" (ref {a_ref:.1f})" if np.isfinite(a_ref) else ""
    ax.set_title(f"{g.TARGETID}\na {g.a_bar:.1f}{ref}  W {g.W_bar:.1f}″  L {g.L_kpc:.1f} kpc", fontsize=7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="Table with TARGETID and a_ref, to draw it")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cutouts", type=Path, default=Path("outputs/cutouts"))
    p.add_argument("--survey", choices=["desi", "sdss"], default="desi")
    p.add_argument("--per-page", type=int, default=20)
    p.add_argument("--rotate", action="store_true", help="Bar on the x axis")
    p.add_argument("--cmap", default="gray_r", help="e.g. viridis")
    args = p.parse_args()

    bars = pd.read_parquet(args.bars)
    if args.reference:
        bars = bars.merge(pd.read_parquet(args.reference)[["TARGETID", "a_ref"]], on="TARGETID", how="left")
    bars = bars.sort_values("a_bar", ascending=False).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ncol = 5
    for page, start in enumerate(range(0, len(bars), args.per_page), 1):
        chunk = bars.iloc[start:start + args.per_page]
        nrow = int(np.ceil(len(chunk) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.9 * nrow), squeeze=False)
        for ax, g in zip(axes.flat, chunk.itertuples()):
            panel(ax, g, args.cutouts, args.survey, args.rotate, args.cmap)
        for ax in axes.flat[len(chunk):]:
            ax.axis("off")
        fig.suptitle("purple: measured bar (isophote length and PA, Fourier width)   dashed circle: reference a",
                     fontsize=9)
        fig.tight_layout()
        tag = "_x" if args.rotate else ""
        fig.savefig(args.out_dir / f"gallery_{args.survey}{tag}_{page:02d}.png", dpi=110)
        plt.close(fig)
    print(f"{page} pages in {args.out_dir}")


if __name__ == "__main__":
    main()
