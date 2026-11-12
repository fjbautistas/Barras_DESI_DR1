"""
Decomposition control figure, one row per galaxy:
image with the bars (purple: isophote length + Fourier width; green: bar of
the 2D decomposition; dashed circle: reference a, if given), residual
(image - model)/σ, and image minus bulge, disk and ring, where only the bar
should be left.

Usage:
    python tools/plot_decomp.py --bars outputs/bars/bars_common_desi.parquet \\
        --decomp outputs/bars/decomp_known10.parquet --out outputs/gallery/decomp_known10 \\
        --reference outputs/sample/common_valeria.parquet
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checks import decompose
from src.cutout import cutout_paths, pixscale
from tools.plot_gallery import boxy


def row_panels(axes, g, d, cut_dir, survey, a_ref):
    ps = pixscale(survey)
    name = str(g.TARGETID)
    img = fits.getdata(cutout_paths(cut_dir, name, survey)["fits"]).astype(float) - g.sky
    model = fits.getdata(cut_dir / f"{name}_{survey}_model.fits")
    ivar = fits.getdata(cut_dir / f"{name}_{survey}_r_ivar.fits", 1)
    psf = fits.getdata(cut_dir / f"{name}_{survey}_r_psf.fits").astype(float)
    # parquet columns "bar_PA" → fitted parameter names "bar:PA"
    p = {k.replace("_", ":", 1) if k.split("_")[0] in ("bulge", "disk", "bar", "ring") else k: v
         for k, v in d._asdict().items()}
    bar_img = decompose.component_image(p, "bar", psf, img.shape)
    x0, y0 = d.X0 - 1, d.Y0 - 1
    h = 2.2 * np.nanmax([g.a_eps, d.bar_a, a_ref or 0]) / ps
    norm = ImageNormalize(img, interval=PercentileInterval(99.8), stretch=AsinhStretch(0.03))

    green = boxy(d.bar_a / ps, d.bar_b / ps, d.bar_PA + 90, x0, y0, c=d.bar_c0 + 2)
    purple = boxy(g.a_eps / ps, g.W_comb / 2 / ps, g.pa_bar + 90, x0, y0)
    axes[0].imshow(img, origin="lower", cmap="gray_r", norm=norm)
    axes[0].plot(*purple, color="tab:purple", lw=1.4)
    axes[0].plot(*green, color="tab:green", lw=1.2)
    if a_ref:
        axes[0].add_patch(Circle((x0, y0), a_ref / ps, fill=False, color="k", ls="--", lw=0.8))
    axes[1].imshow((img - model) * np.sqrt(np.clip(ivar, 0, None)), origin="lower",
                   cmap="RdBu_r", vmin=-10, vmax=10)
    axes[2].imshow(img - model + bar_img, origin="lower", cmap="gray_r", norm=norm)
    axes[2].plot(*green, color="tab:green", lw=1.2)
    axes[2].plot(*purple, color="tab:purple", lw=1.0)
    ref = f"  ref {a_ref:.1f}″" if a_ref else ""
    axes[0].set_title(f"{name}\na: iso {g.a_eps:.1f}  2D {d.bar_a:.1f}{ref}   PA: iso {g.pa_bar:.0f}  2D {d.bar_PA % 180:.0f}",
                      fontsize=7)
    axes[1].set_title(f"residual/σ   χ²r = {d.chi2r:.2f}", fontsize=7)
    axes[2].set_title(f"image − (bulge+disk+ring)   ring R = {d.ring_R:.1f}″", fontsize=7)
    for ax in axes:
        ax.set_xlim(x0 - h, x0 + h), ax.set_ylim(y0 - h, y0 + h)
        ax.set_xticks([]), ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=Path, required=True)
    p.add_argument("--decomp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output stem; pages get _01.png, …")
    p.add_argument("--reference", type=Path)
    p.add_argument("--cutouts", type=Path, default=Path("outputs/cutouts"))
    p.add_argument("--survey", choices=["desi", "sdss"], default="desi")
    p.add_argument("--per-page", type=int, default=5)
    args = p.parse_args()

    bars = pd.read_parquet(args.bars).set_index("TARGETID")
    dec = pd.read_parquet(args.decomp)
    ref = (pd.read_parquet(args.reference).set_index("TARGETID")["a_ref"]
           if args.reference else pd.Series(dtype=float))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(dec), args.per_page), 1):
        chunk = dec.iloc[start:start + args.per_page]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(9, 3.1 * len(chunk)), squeeze=False)
        for ax_row, d in zip(axes, chunk.itertuples(index=False)):
            g = next(bars.loc[[d.TARGETID]].reset_index().itertuples())
            row_panels(ax_row, g, d, args.cutouts, args.survey, ref.get(d.TARGETID))
        fig.suptitle("purple: isophote length + Fourier width   green: 2D decomposition bar   dashed: reference",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{args.out}_{page:02d}.png", dpi=105)
        plt.close(fig)
    print(page, "pages")


if __name__ == "__main__":
    main()
