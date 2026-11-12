"""
Our bars against the reference measurements, split into the galaxies of
the DESI parent sample (used while building the method) and the new ones
(never looked at): median ratio, half 16–84 % range and fraction within 20 %
for the semi-length a_bar, the width W_bar and the isophote width 2 b_iso.

Usage:
    python tools/compare_reference.py --bars outputs/bars/bars_reference_desi.parquet \\
        --reference outputs/sample/reference_all.parquet --out outputs/gallery/reference_1to1.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = [("in DESI (157, used to build)", True, "tab:blue"),
          ("new (291, never looked at)", False, "tab:purple")]


def stats(x, ref):
    ok = np.isfinite(x) & np.isfinite(ref) & (ref > 0)
    q = x[ok] / ref[ok]
    p16, p50, p84 = np.percentile(q, [16, 50, 84])
    return {"N": int(ok.sum()), "median": p50, "half_68": (p84 - p16) / 2,
            "within_20": float(np.mean(np.abs(q - 1) < 0.2))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = pd.read_parquet(args.bars).merge(
        pd.read_parquet(args.reference)[["TARGETID", "a_ref", "b_ref", "in_desi"]], on="TARGETID")
    quantities = [("semi-length a", "a_bar", lambda g: g.a_ref),
                  ("width W (Fourier, no bulge)", "W_bar", lambda g: 2 * g.b_ref),
                  ("width W (isophote, with bulge)", "W_iso", lambda g: 2 * g.b_ref)]
    d["W_iso"] = 2 * d["b_iso"]

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (label, col, ref_of) in zip(axes, quantities):
        lim = np.nanpercentile(np.r_[d[col], ref_of(d)], 99) * 1.1
        for name, flag, color in GROUPS:
            g = d[d.in_desi == flag]
            s = stats(g[col].to_numpy(), ref_of(g).to_numpy())
            rows.append({"quantity": label, "group": name, **s})
            ax.scatter(ref_of(g), g[col], s=10, alpha=0.6, color=color,
                       label=f"{name}: {s['median']:.2f} ±{s['half_68']:.2f}, {100 * s['within_20']:.0f} % in 20 %")
        ax.plot([0, lim], [0, lim], color="k", lw=1.5)
        ax.set_xlim(0, lim), ax.set_ylim(0, lim), ax.set_aspect("equal")
        ax.set_xlabel(f"reference {label} [″]"), ax.set_ylabel(f"this work {label} [″]")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    table = pd.DataFrame(rows)
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
