"""Figura: comparación 1:1 con las 448 barras de Valeria (proyectado).
(a) semilargo: isofotas; (b) ancho: isofotas (2b, con bulbo) y combinada (Fourier, sin bulbo);
(c) largo en kpc. Puntos llenos: las 157 con las que se ajustó el método; vacíos: las 291 que no
se usaron para ajustarlo (prueba con referencia)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _estilo as e

d = pd.read_parquet(e.DATA / "bars_reference_desi.parquet").merge(
    pd.read_parquet(e.DATA / "reference_all.parquet")[["TARGETID", "a_ref", "b_ref", "in_desi", "kpc_per_arcsec_ref"]],
    on="TARGETID")
d["L_ref_kpc"] = 2 * d.a_ref * d.kpc_per_arcsec_ref
rows = []
for lab, x, ref in [("a isofotas", d.a_eps, d.a_ref), ("a Fourier", d.a_fourier, d.a_ref),
                    ("W isofotas", 2 * d.b_bar, 2 * d.b_ref), ("W combinada", d.W_comb, 2 * d.b_ref),
                    ("L kpc", d.L_proj_kpc, d.L_ref_kpc)]:
    for grp, m in [("157", d.in_desi), ("291", ~d.in_desi), ("448", d.in_desi | ~d.in_desi)]:
        n, med, sig, f20 = e.ratio_stats(x[m].to_numpy(), ref[m].to_numpy())
        rows.append((lab, grp, n, med, sig, f20))
tab = pd.DataFrame(rows, columns=["cantidad", "grupo", "N", "mediana", "semiancho", "frac<20%"])
print(tab.round(2).to_string(index=False))
print("kpc/″ nuestro / Valeria: mediana", (d.kpc_per_arcsec / d.kpc_per_arcsec_ref).median().round(3))
print("muy alejadas (a fuera de 0.6-1.5):", ((d.a_eps / d.a_ref < 0.6) | (d.a_eps / d.a_ref > 1.5)).sum())

e.cm.style(e.W2)
fig, ax = plt.subplots(1, 3, figsize=(e.W2, 0.36 * e.W2), layout="constrained")
panels = [(ax[0], [(d.a_eps, d.a_ref, e.ISO, "isofotas")], r"semilargo $a$ [$^{\prime\prime}$]", 22),
          (ax[1], [(2 * d.b_bar, 2 * d.b_ref, e.ISO, "isofotas"), (d.W_comb, 2 * d.b_ref, e.COMB, "combinada")],
           r"ancho $W$ [$^{\prime\prime}$]", 16),
          (ax[2], [(d.L_proj_kpc, d.L_ref_kpc, e.ISO, "isofotas")], r"largo $L$ [kpc]", 40)]
for a_, series, lab, lim in panels:
    for x, ref, col, name in series:
        a_.scatter(ref[d.in_desi], x[d.in_desi], s=4, color=col, lw=0, label=f"{name}, 157")
        a_.scatter(ref[~d.in_desi], x[~d.in_desi], s=5, facecolors="none", edgecolors=col, lw=0.4,
                   label=f"{name}, 291")
    a_.plot([0, lim], [0, lim], color="k", lw=0.8)
    a_.set_xlim(0, lim), a_.set_ylim(0, lim), a_.set_aspect("equal")
    a_.set_xlabel(f"Valeria: {lab}"), a_.set_ylabel(f"este trabajo: {lab}")
    a_.legend(fontsize=5, frameon=False, loc="upper left")
for a_, t in zip(ax, "abc"):
    a_.set_title(f"({t})", fontsize=6.5)
e.save(fig, "valeria")
tab.to_csv(e.OUT / "valeria_tabla.csv", index=False)
