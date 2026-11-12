"""
Bar semi-length a and full width W of both methods against the reference
measurements (projected, arcsec), with the median ratio and its scatter.

Usage:
    python tools/plot_vs_reference.py --bars outputs/bars/bars_common_desi.parquet \\
        --reference outputs/sample/common_valeria.parquet --out outputs/gallery/vs_reference_desi.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stats(x, ref):
    ok = np.isfinite(x) & np.isfinite(ref)
    q = x[ok] / ref[ok]
    p16, p50, p84 = np.percentile(q, [16, 50, 84])
    return ok.sum(), p50, (p84 - p16) / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = pd.read_parquet(args.bars).merge(
        pd.read_parquet(args.reference)[["TARGETID", "a_ref", "b_ref"]], on="TARGETID")
    panels = [("semi-length a [″]", d.a_ref, [("isophotes", d.a_eps, "tab:blue"),
                                               ("Fourier", d.a_fourier, "tab:orange")]),
              ("width W [″]", 2 * d.b_ref, [("isophotes", 2 * d.b_bar, "tab:blue"),
                                             ("Fourier", d.W_fourier, "tab:orange"),
                                             ("combined", d.W_comb, "tab:purple")])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (label, ref, series) in zip(axes, panels):
        for name, x, col in series:
            n, med, sig = stats(x.to_numpy(), ref.to_numpy())
            ax.scatter(ref, x, s=12, color=col, alpha=0.7,
                       label=f"{name}: N={n}, median ratio {med:.2f}, ±{sig:.2f}")
        lim = np.nanpercentile(np.r_[ref, *[s[1] for s in series]], 99) * 1.1
        ax.plot([0, lim], [0, lim], color="k", lw=1.5)
        ax.set_xlim(0, lim), ax.set_ylim(0, lim), ax.set_aspect("equal")
        ax.set_xlabel(f"reference {label}"), ax.set_ylabel(f"this work {label}")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(args.out)


if __name__ == "__main__":
    main()
