"""
The 24 galaxies whose bar did not match when Valeria applied the SDSS-trained
method to DESI (modelo/no_calzan), against the rest of the 448.

Reference: a_arcsec of the training table, the full bar length of Alonso et al.
(2024) in SDSS (L_Valeria = 2 a_bar_arcsec agrees with it, ratio 1.06). Ours:
L = 2 a_bar in SDSS and in DESI FITS. Two of the 24 are not in the training
table; they were measured apart (outputs/sample/no_calzan_extra.parquet).

Usage:
    python checks/no_calzan.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "outputs/bars"


def coords(ra, dec):
    return SkyCoord(np.asarray(ra, float) * u.deg, np.asarray(dec, float) * u.deg)


def stats(q):
    q = q[np.isfinite(q)]
    p16, p50, p84 = np.percentile(q, [16, 50, 84])
    return f"{p50:.2f} ±{(p84 - p16) / 2:.2f}, {100 * np.mean(np.abs(q - 1) < 0.2):.0f} % en 20 % (N = {len(q)})"


def main():
    tr = pd.read_csv(ROOT / "modelo/entrenamiento_parametros")
    nc = pd.read_csv(ROOT / "modelo/no_calzan")
    ref = pd.read_parquet(ROOT / "outputs/sample/reference_all.parquet")
    ours = {s: pd.concat([pd.read_parquet(B / f"bars_reference_{s}.parquet"),
                          pd.read_parquet(B / f"bars_nocalzan_extra_{s}.parquet")]).set_index("TARGETID")
            for s in ("desi", "sdss")}

    # the 448: paper length from the training table, joined by its id (id_ref = id_desi)
    t = tr.set_index("id_desi").loc[ref.id_ref]
    d = pd.DataFrame({"TARGETID": ref.TARGETID, "L_ref": t.a_arcsec.values,
                      "L_val": t.length_bar_arcsec.values, "Z": ref.Z})
    extra = pd.read_parquet(ROOT / "outputs/sample/no_calzan_extra.parquet")
    k = [nc.index[np.argmin(coords(nc.ra, nc.dec).separation(coords([r.RA], [r.DEC])[0]).arcsec)]
         for r in extra.itertuples()]
    d = pd.concat([d, pd.DataFrame({"TARGETID": extra.TARGETID, "L_ref": nc.a_arcsec.values[k],
                                    "L_val": np.nan, "Z": extra.Z})], ignore_index=True)
    j, sep_nc, _ = coords(nc.ra, nc.dec).match_to_catalog_sky(
        coords(np.r_[ref.RA, extra.RA], np.r_[ref.DEC, extra.DEC]))
    assert sep_nc.arcsec.max() < 1
    d["no_calzan"] = False
    d.loc[j, "no_calzan"] = True
    for s, t in ours.items():
        d[f"L_{s}"] = 2 * t.loc[d.TARGETID, "a_bar"].values
        d[f"flag_{s}"] = (t.loc[d.TARGETID, ["flag_eps", "flag_end", "flag_pa"]].all(axis=1)).values

    print(f"no calzan: {d.no_calzan.sum()}   resto: {(~d.no_calzan).sum()}")
    for name, g in (("no calzan", d[d.no_calzan]), ("resto", d[~d.no_calzan])):
        print(f"\n{name}")
        print("  Valeria SDSS / paper :", stats((g.L_val / g.L_ref).to_numpy()))
        print("  nuestro SDSS / paper :", stats((g.L_sdss / g.L_ref).to_numpy()))
        print("  nuestro DESI / paper :", stats((g.L_desi / g.L_ref).to_numpy()))
        print("  nuestro DESI / SDSS  :", stats((g.L_desi / g.L_sdss).to_numpy()))
    cols = ["TARGETID", "Z", "L_ref", "L_val", "L_sdss", "L_desi", "flag_sdss", "flag_desi"]
    t = d.loc[d.no_calzan, cols].copy()
    t["sdss/ref"], t["desi/ref"] = t.L_sdss / t.L_ref, t.L_desi / t.L_ref
    print("\n" + t.round(2).sort_values("desi/ref").to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, s in zip(axes, ("sdss", "desi")):
        for flag, color, lab in ((False, "tab:blue", "resto"), (True, "tab:purple", "no calzan")):
            g = d[d.no_calzan == flag]
            ax.scatter(g.L_ref, g[f"L_{s}"], s=12 if not flag else 28, color=color, alpha=0.5 if not flag else 0.9,
                       label=f"{lab}: {stats((g[f'L_{s}'] / g.L_ref).to_numpy())}")
        lim = 45
        x = np.array([0, lim])
        ax.plot(x, x, color="k", lw=1.5)
        ax.fill_between(x, 0.8 * x, 1.2 * x, color="0.9", zorder=0)
        ax.set_xlim(0, lim), ax.set_ylim(0, lim), ax.set_aspect("equal")
        ax.set_xlabel("largo del paper (SDSS) [″]"), ax.set_ylabel(f"nuestro largo 2a, {s.upper()} [″]")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = ROOT / "outputs/gallery/no_calzan_1to1.png"
    fig.savefig(out, dpi=130)
    t.to_csv(out.with_suffix(".csv"), index=False)
    print(out)


if __name__ == "__main__":
    main()
