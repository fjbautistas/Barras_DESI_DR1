"""Figura: el método de Alonso et al. (2024) aplicado a la galaxia ejemplo.
(a) imagen girada con la barra sobre x, los cuatro puntos A, B, C, D y el contorno de isobrillo;
(b) perfil a lo largo de x; (c) perfil a lo largo de y. Los puntos se ponen en los extremos que midió
Valeria (a = 14.05″, b = 4.83″); el nivel es el promedio de los cuatro brillos."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import _estilo as e

B_REF = 4.83
g = e.galaxy(e.EXAMPLE)
ps = g["ps"]
rot, cx, cy = e.on_x(g["img"], g["x0"], g["y0"], g["b"]["pa_bar"])
sm = gaussian_filter(rot, 1.5)           # suavizado, como el desenfoque del paper
ic, jc = int(round(cy)), int(round(cx))
px = sm[ic, :]
py = sm[:, jc]
xa = (np.arange(sm.shape[1]) - cx) * ps
ya = (np.arange(sm.shape[0]) - cy) * ps
iA, iB = int(round(cx - e.A_REF_EXAMPLE / ps)), int(round(cx + e.A_REF_EXAMPLE / ps))
iC, iD = int(round(cy - B_REF / ps)), int(round(cy + B_REF / ps))
vals = [px[iA], px[iB], py[iC], py[iD]]
level = np.mean(vals)
print("brillos A, B, C, D [nmgy/px]:", np.round(vals, 4), "| nivel promedio:", round(level, 4))

# extensión del contorno a lo largo de los ejes
above_x = xa[px >= level]
above_y = ya[py >= level]
print(f"contorno: semieje x = {(above_x.max() - above_x.min()) / 2:.2f}″, semieje y = {(above_y.max() - above_y.min()) / 2:.2f}″")

e.cm.style(e.W2)
fig = plt.figure(figsize=(e.W2, 0.36 * e.W2), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
a = fig.add_subplot(gs[:, 0])
h = 22 / ps
a.imshow(sm, origin="lower", cmap="gray_r", norm=e.asinh_norm(sm))
a.contour(sm, levels=[level], colors=[e.FOU], linewidths=1.0)
for (x, y, lab) in [(iA, ic, "A"), (iB, ic, "B"), (jc, iC, "C"), (jc, iD, "D")]:
    a.plot(x, y, "o", ms=3, color=e.REF)
    a.annotate(lab, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
e.zoom(a, cx, cy, h)
a.set_title("(a) barra sobre $x$ y contorno", fontsize=6.5)

for k, (prof, ax_, pts, labs, name) in enumerate(
        [(px, xa, [iA, iB], "AB", "x"), (py, ya, [iC, iD], "CD", "y")]):
    s = fig.add_subplot(gs[k, 1])
    s.semilogy(ax_, prof, color=e.ISO, lw=0.8)
    s.axhline(level, color=e.FOU, lw=0.8, ls="--", label="nivel promedio")
    for p_, lab in zip(pts, labs):
        s.plot(ax_[p_], prof[p_], "o", ms=3, color=e.REF)
        s.annotate(lab, (ax_[p_], prof[p_]), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    s.set_xlim(-22, 22)
    win = np.abs(ax_) <= 22
    s.set_ylim(0.7 * prof[win].min(), 1.6 * prof[win].max())
    s.set_ylabel(f"brillo en ${name}$")
    if k == 0:
        s.legend(loc="lower center", frameon=False)
        s.set_title("(b) perfiles por el centro", fontsize=6.5)
    else:
        s.set_xlabel(r"distancia al centro [$^{\prime\prime}$]")
e.save(fig, "paper")
