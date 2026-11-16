"""Figura: solo r frente a solo z en las 448 (mismo método). (a) z/Valeria contra r/Valeria;
(b) fracción a menos del 20 % por tamaño de la barra; (c) por redshift. Imprime las cifras del texto, incluido el flag |r/z-1|."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _estilo as e

g = pd.read_csv(e.DATA / "grz_448.csv").pivot(index="TARGETID", columns="metodo", values="a")
ref = pd.read_parquet(e.DATA / "reference_all.parquet").set_index("TARGETID")[["a_ref", "Z"]]
d = g[["r", "z"]].join(ref)
for b in "rz":
    d["q" + b] = d[b] / d.a_ref
    d["ok" + b] = (d["q" + b] - 1).abs() < 0.2
for b in "rz":
    q = d["q" + b]
    print(f"{b}: <20% {d['ok' + b].mean():.3f}  <10% {((q - 1).abs() < .1).mean():.3f}  med {q.median():.3f}  "
          f"semiancho {(q.quantile(.84) - q.quantile(.16)) / 2:.3f}")
print(f"z frente a r: arregla {(d.okz & ~d.okr).sum()}, rompe {(~d.okz & d.okr).sum()}")
fl = (d.r / d.z - 1).abs() > 0.15
print(f"flag |r/z-1|>0.15: marca {fl.sum()}, fallan en r {(~d.okr)[fl].sum()} ({(~d.okr)[fl].mean():.0%}); "
      f"en toda la muestra fallan {(~d.okr).mean():.0%}")
TAB = {}
for col, bins in [("a_ref", [0, 4, 6, 9, 30]), ("Z", [0, .03, .05, .07, .2])]:
    t = d.groupby(pd.cut(d[col], bins), observed=True)[["okr", "okz"]].agg(["mean", "size"])
    TAB[col] = t
    print(col); print(t.round(3).to_string())

e.cm.style(e.W2)
fig, ax = plt.subplots(1, 3, figsize=(e.W2, 0.34 * e.W2), layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
ax[0].scatter(d.qr, d.qz, s=4, color="0.4", lw=0, alpha=0.6)
ax[0].plot([0, 3], [0, 3], color="k", lw=0.6)
for v in (0.8, 1.2):
    ax[0].axhline(v, color="0.6", lw=0.5, ls=":"), ax[0].axvline(v, color="0.6", lw=0.5, ls=":")
ax[0].set_xlim(0, 2.5), ax[0].set_ylim(0, 2.5)
ax[0].set_xlabel(r"$r$ / Valeria"), ax[0].set_ylabel(r"$z$ / Valeria"), ax[0].set_title("(a)", fontsize=6.5, loc="left")
for s, (col, lab, tag) in zip(ax[1:], [("a_ref", r"semilargo de Valeria [$^{\prime\prime}$]", "(b)"), ("Z", "redshift", "(c)")]):
    t = TAB[col]; x = np.arange(len(t))
    for b, c, dx in [("r", e.COMB, -0.08), ("z", e.FOU, 0.08)]:
        f, n = t[("ok" + b, "mean")].to_numpy(), t[("ok" + b, "size")].to_numpy()
        s.errorbar(x + dx, f, np.sqrt(f * (1 - f) / n), fmt="o-", color=c, ms=3, lw=1, capsize=2, label=f"${b}$")
    s.set_xticks(x), s.set_xticklabels([f"{iv.left:g}–{iv.right:g}" for iv in t.index], fontsize=5.5)
    s.set_ylim(0.7, 1.0), s.set_xlabel(lab), s.set_title(tag, fontsize=6.5, loc="left")
ax[1].set_ylabel(r"fracción a menos del 20\,\%"), ax[1].legend(frameon=False, fontsize=6)
e.save(fig, "r_z")
