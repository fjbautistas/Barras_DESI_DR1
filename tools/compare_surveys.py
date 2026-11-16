"""
Method vs data: our measurements on the same SDSS images as the reference
(method only), on DESI vs SDSS (data only) and on DESI vs the reference
(both). Median ratio, half 16–84 % range and fraction within 20 % for the
semi-length a_bar, the Fourier width W_bar and the isophote width 2 b_iso.

Usage:
    python tools/compare_surveys.py --desi outputs/bars/bars_reference_desi.parquet \\
        --sdss outputs/bars/bars_reference_sdss.parquet \\
        --reference outputs/sample/reference_all.parquet --out outputs/gallery/method_vs_data.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_reference import stats

COLS = ["a_bar", "W_bar", "W_iso"]
CASES = [("method: SDSS ours / reference", "sdss", "ref", "tab:blue"),
         ("data: DESI ours / SDSS ours", "desi", "sdss", "tab:purple"),
         ("total: DESI ours / reference", "desi", "ref", "tab:orange")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--desi", type=Path, required=True)
    p.add_argument("--sdss", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tabs = {}
    for name, path in (("desi", args.desi), ("sdss", args.sdss)):
        t = pd.read_parquet(path)
        t["W_iso"] = 2 * t["b_iso"]
        tabs[name] = t.set_index("TARGETID")[COLS]
    r = pd.read_parquet(args.reference).set_index("TARGETID")
    tabs["ref"] = pd.DataFrame({"a_bar": r.a_ref, "W_bar": 2 * r.b_ref, "W_iso": 2 * r.b_ref})
    ids = tabs["ref"].index.intersection(tabs["desi"].index).intersection(tabs["sdss"].index)

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (label, num, den, color) in zip(axes, CASES):
        x, y = tabs[den].loc[ids, "a_bar"], tabs[num].loc[ids, "a_bar"]
        for col in COLS:
            rows.append({"case": label, "quantity": col,
                         **stats(tabs[num].loc[ids, col].to_numpy(), tabs[den].loc[ids, col].to_numpy())})
        s = rows[-3]
        lim = np.nanpercentile(np.r_[x, y], 99) * 1.1
        ax.scatter(x, y, s=10, alpha=0.6, color=color)
        ax.plot([0, lim], [0, lim], color="k", lw=1.5)
        ax.set_xlim(0, lim), ax.set_ylim(0, lim), ax.set_aspect("equal")
        ax.set_xlabel(f"{den} a [″]"), ax.set_ylabel(f"{num} a [″]")
        ax.set_title(f"{label}\n{s['median']:.2f} ±{s['half_68']:.2f}, "
                     f"{100 * s['within_20']:.0f} % in 20 %  (N = {s['N']})", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    table = pd.DataFrame(rows)
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
