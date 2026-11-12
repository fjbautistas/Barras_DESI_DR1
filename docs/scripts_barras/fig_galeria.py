"""Figura: galería con la barra sobre el eje x. Fila de arriba: 5 de las 291 galaxias de Valeria que no
se usaron para ajustar el método (al azar, semilla 3); fila de abajo: 5 de las 10 galaxias ciegas de
DESI DR1 (sin referencia). Morado: barra combinada (largo isofotas, ancho Fourier, forma ilustrativa
|x/a|^3 + |y/b|^3 = 1); círculo negro a trazos: semilargo de Valeria."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.patches import Circle

import _estilo as e

ref = pd.read_parquet(e.DATA / "reference_all.parquet")[["TARGETID", "a_ref", "b_ref", "in_desi"]]
new = pd.read_parquet(e.DATA / "bars_reference_desi.parquet").merge(ref, on="TARGETID")
new = new[~new.in_desi].sample(5, random_state=3)
blind = pd.read_parquet(e.DATA / "bars_blind10_desi.parquet").sample(5, random_state=3)
blind["a_ref"] = np.nan


def panel(ax, g):
    ps = 0.262
    img = fits.getdata(e.cutout_paths(e.CUT, str(g.TARGETID), "desi")["fits"]).astype(float) - g.sky
    rot, cx, cy = e.on_x(img, g.x0, g.y0, g.pa_bar + 90)
    ax.imshow(rot, origin="lower", cmap="gray_r", norm=e.asinh_norm(rot))
    ax.plot(*e.boxy(g.a_eps / ps, g.W_comb / 2 / ps, 0, cx, cy), color=e.COMB, lw=1)
    if np.isfinite(g.a_ref):
        ax.add_patch(Circle((cx, cy), g.a_ref / ps, fill=False, color=e.REF, ls="--", lw=0.6))
        txt = f"$a = {g.a_eps:.1f}''$ (ref ${g.a_ref:.1f}''$)"
    else:
        txt = f"$a = {g.a_eps:.1f}''$"
    e.zoom(ax, cx, cy, 2.0 * np.nanmax([g.a_eps, g.a_ref]) / ps)
    ax.set_title(txt, fontsize=5.5)
    print(g.TARGETID, txt)


e.cm.style(e.W2)
fig, ax = plt.subplots(2, 5, figsize=(e.W2, 0.43 * e.W2), layout="constrained")
for a_, g in zip(ax[0], new.itertuples()):
    panel(a_, g)
for a_, g in zip(ax[1], blind.itertuples()):
    panel(a_, g)
ax[0, 0].set_ylabel("Valeria, no usadas", fontsize=6)
ax[1, 0].set_ylabel("ciegas DESI", fontsize=6)
e.save(fig, "galeria")
