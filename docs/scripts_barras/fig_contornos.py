"""Figura: por qué combinar. Galaxia ejemplo con la barra sobre x.
(a) isofota real al nivel de a_ε (una S: barra + arranque de los brazos) y la elipse que la resume;
(b) contorno a media altura de la imagen solo-barra; (c) barras dibujadas con los parámetros de cada
método: isofotas (L y ancho de la elipse), Fourier y combinada (L isofotas, ancho Fourier)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import _estilo as e

g = e.galaxy(e.EXAMPLE)
ps, img, x0, y0, b, t = g["ps"], g["img"], g["x0"], g["y0"], g["b"], g["t"]
f = e.fourier_of(g)
fb = f["fb"]
rot, cx, cy = e.on_x(img, x0, y0, b["pa_bar"])
lev = float(t["intens"][np.argmin(np.abs(t["sma"] * ps - b["a_eps"]))])
bo = e.fourier.bar_only_image(img.shape, x0, y0, f["r"], f["am"], f["bm"], b["a_eps"] / ps)
bo, _, _ = e.on_x(bo, x0, y0, b["pa_bar"])
xs = [int(round(cx + s * k * b["a_eps"] / ps)) for k in (0.5, 0.75) for s in (-1, 1)]
lev_bo = 0.5 * np.mean([bo[int(round(cy)) - 3:int(round(cy)) + 4, x].max() for x in xs])
print(f"isofotas: L = {2 * b['a_eps']:.1f}″ W = {2 * b['b_bar']:.1f}″ | Fourier: L = {2 * fb['a_phase'] * ps:.1f}″ "
      f"W = {f['w'] * ps:.1f}″ | combinada: L = {2 * b['a_eps']:.1f}″ W = {f['wc'] * ps:.1f}″")

e.cm.style(e.W2)
fig, ax = plt.subplots(1, 3, figsize=(e.W2, 0.36 * e.W2), layout="constrained")
norm = e.asinh_norm(rot)
for a_ in ax:
    a_.imshow(rot, origin="lower", cmap="gray_r", norm=norm)
ax[0].contour(rot, levels=[lev], colors=[e.ISO], linewidths=0.9)
ax[0].plot(*e.boxy(b["a_eps"] / ps, b["b_bar"] / ps, 0, cx, cy, c=2), color=e.ISO, lw=0.8, ls="--")
ax[0].set_title(r"(a) isofota en $a_\varepsilon$ y su elipse", fontsize=6.5)
ax[1].contour(bo, levels=[lev_bo], colors=[e.FOU], linewidths=0.9)
ax[1].set_title("(b) contorno de la imagen solo-barra", fontsize=6.5)
dpa = fb["phi2_bar"] - b["pa_bar"]
ax[2].plot(*e.boxy(b["a_eps"] / ps, b["b_bar"] / ps, 0, cx, cy), color=e.ISO, lw=1)
ax[2].plot(*e.boxy(fb["a_phase"], f["w"] / 2, dpa, cx, cy), color=e.FOU, lw=1)
ax[2].plot(*e.boxy(b["a_eps"] / ps, f["wc"] / 2, 0, cx, cy), color=e.COMB, lw=1.3)
for s in (-1, 1):
    ax[2].plot(cx + s * e.A_REF_EXAMPLE / ps, cy, "o", ms=3, mfc="none", mec=e.REF)
ax[2].legend([Line2D([], [], color=c) for c in (e.ISO, e.FOU, e.COMB)] +
             [Line2D([], [], ls="", marker="o", mfc="none", mec=e.REF)],
             ["isofotas", "Fourier", "combinada", "extremos de Valeria"], fontsize=5, frameon=False,
             loc="lower center", ncol=2)
ax[2].set_title("(c) barras de cada método", fontsize=6.5)
for a_ in ax:
    e.zoom(a_, cx, cy, 22 / ps)
e.save(fig, "contornos")
