"""Figura: cómo mejoró el largo por isofotas con cada corrección (las 157 en común con Valeria).
v0 primer código; v1 centro cerca del recorte + ajuste radio por radio + primer pico de ε;
v2 + código 2 sin moverse descartado (solo ε y PA iguales) y ancho combinado; v3 + inicio en los
momentos de la luz y ε sin moverse; v4 + PA promediado sobre la barra.
(a) mediana y semiancho 16-84 % del cociente a/a_ref; (b) fracción a menos del 20 % y número de
galaxias fuera de 0.6-1.5."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _estilo as e

ref = pd.read_parquet(e.DATA / "reference_all.parquet")[["TARGETID", "a_ref"]]
labels = ["v0", "v1", "v2", "v3", "v4"]
rows = []
for v in labels:
    d = pd.read_parquet(e.DATA / f"bars_common_desi_{v}.parquet").merge(ref, on="TARGETID")
    n, med, sig, f20 = e.ratio_stats(d.a_eps.to_numpy(), d.a_ref.to_numpy())
    q = d.a_eps / d.a_ref
    rows.append((v, n, med, sig, f20, int(((q < 0.6) | (q > 1.5)).sum())))
tab = pd.DataFrame(rows, columns=["versión", "N", "mediana", "semiancho", "frac<20%", "fuera 0.6-1.5"])
print(tab.round(3).to_string(index=False))

e.cm.style(e.W2)
fig, ax = plt.subplots(1, 2, figsize=(e.W2, 0.3 * e.W2), layout="constrained")
x = np.arange(len(labels))
ax[0].errorbar(x, tab.mediana, tab.semiancho, fmt="o", ms=3, color=e.ISO, capsize=2, lw=0.8)
ax[0].axhline(1, color="k", lw=0.6)
ax[0].set_ylabel(r"mediana de $a_\varepsilon / a_{\rm ref}$ ($\pm$ 16--84\,\%)")
ax[1].plot(x, 100 * tab["frac<20%"], "o-", ms=3, color=e.ISO, lw=0.8, label="a menos del 20\\,\\%")
ax2 = ax[1].twinx()
ax2.plot(x, tab["fuera 0.6-1.5"], "s--", ms=3, color=e.FOU, lw=0.8)
ax[1].set_ylabel("galaxias a menos del 20\\,\\% [\\%]", color=e.ISO)
ax2.set_ylabel("fuera de 0.6--1.5 [N]", color=e.FOU)
for a_, t in zip(ax, "ab"):
    a_.set_xticks(x, labels)
    a_.set_title(f"({t})", fontsize=6.5)
e.save(fig, "versiones")
