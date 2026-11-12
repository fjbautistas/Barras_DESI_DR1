"""Figura: Fourier en la galaxia ejemplo.
(a) imagen con dos círculos (r = 8″ sobre la barra, r = 25″ sobre los brazos);
(b, c) brillo a lo largo de cada círculo y su reconstrucción con m = 0, m = 0 + 2 y m = 0..6;
(d) A2/A0(r) y (e) fase φ2(r), con el fin de la barra por la fase; (f) imagen solo-barra (m = 2, 4, 6)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.ndimage import map_coordinates

import _estilo as e

g = e.galaxy(e.EXAMPLE)
ps, img, x0, y0 = g["ps"], g["img"], g["x0"], g["y0"]
f = e.fourier_of(g)
fb = f["fb"]
print(f"a_fase = {fb['a_phase'] * ps:.2f}″  pico A2/A0 = {fb['a2_max']:.2f} en r = {fb['r_a2_max'] * ps:.2f}″  "
      f"φ2(barra) = {fb['phi2_bar']:.1f}°  W_Fourier = {f['w'] * ps:.2f}″  W_comb = {f['wc'] * ps:.2f}″")

phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
m = np.arange(1, 7)
D = np.column_stack([np.ones_like(phi), np.cos(np.outer(phi, m)), np.sin(np.outer(phi, m))])

e.cm.style(e.W2)
fig = plt.figure(figsize=(e.W2, 0.62 * e.W2), layout="constrained")
gs = fig.add_gridspec(2, 3)
a = fig.add_subplot(gs[0, 0])
a.imshow(img, origin="lower", cmap="gray_r", norm=e.asinh_norm(img))
for R, col in [(8, e.ISO), (25, e.COMB)]:
    a.add_patch(Circle((x0, y0), R / ps, fill=False, color=col, lw=1))
e.zoom(a, x0, y0, 30 / ps)
a.set_title("(a) círculos de medida", fontsize=6.5)

for k, (R, col, tag) in enumerate([(8, e.ISO, "(b) $r = 8''$, barra"), (25, e.COMB, "(c) $r = 25''$, brazos")]):
    s = fig.add_subplot(gs[0, k + 1])
    r = R / ps
    v = map_coordinates(img, [y0 + r * np.sin(phi), x0 + r * np.cos(phi)], order=1)
    c = np.linalg.lstsq(D, v, rcond=None)[0]
    deg = np.degrees(phi)
    s.plot(deg, v, ".", ms=1.5, color="0.6", label="píxeles")
    s.axhline(c[0], color="k", lw=0.8, label="$m = 0$")
    s.plot(deg, c[0] + c[2] * np.cos(2 * phi) + c[8] * np.sin(2 * phi), color=e.FOU, lw=1.0,
           label=f"$m = 0 + 2$ ($A_2/A_0 = {np.hypot(c[2], c[8]) / c[0]:.2f}$)")
    s.plot(deg, D @ c, color=col, lw=0.8, label="$m = 0..6$")
    s.set_xticks(range(0, 361, 90))
    s.set_xlabel(r"ángulo $\varphi$ [$^\circ$]")
    s.set_title(tag, fontsize=6.5)
    s.legend(fontsize=5, frameon=False, loc="upper right")
    print(f"r = {R}″: A2/A0 = {np.hypot(c[2], c[8]) / c[0]:.2f}, fase m=2 = {np.degrees(np.arctan2(c[8], c[2])) / 2 % 180:.1f}°")

ra = f["r"] * ps
s = fig.add_subplot(gs[1, 0])
s.plot(ra, fb["a2"], color=e.FOU, lw=0.9)
s.set_ylabel("$A_2/A_0$")
s.set_xlabel(r"$r$ [$^{\prime\prime}$]")
s.set_title("(d) amplitud de $m = 2$", fontsize=6.5)
s2 = fig.add_subplot(gs[1, 1])
s2.plot(ra, fb["phi2"], ".", ms=1.5, color=e.FOU)
s2.axhspan(fb["phi2_bar"] - 10, fb["phi2_bar"] + 10, color=e.FOU, alpha=0.15, lw=0)
s2.set_ylabel(r"$\varphi_2$ [$^\circ$]")
s2.set_xlabel(r"$r$ [$^{\prime\prime}$]")
s2.set_title(r"(e) fase de $m = 2$ ($\pm 10^\circ$ sombreado)", fontsize=6.5)
for s_ in (s, s2):
    s_.set_xlim(0, 40)
    s_.axvspan(0, e.A_MIN, color="0.9", lw=0)
    s_.axvline(fb["a_phase"] * ps, color=e.FOU, lw=0.8)
    s_.axvline(g["b"]["a_eps"], color=e.ISO, lw=0.8)
    s_.axvline(e.A_REF_EXAMPLE, color=e.REF, lw=0.7, ls="--")

bo = e.fourier.bar_only_image(img.shape, x0, y0, f["r"], f["am"], f["bm"], fb["a_phase"])
bo, cx, cy = e.on_x(bo, x0, y0, fb["phi2_bar"])
s3 = fig.add_subplot(gs[1, 2])
lim = np.percentile(np.abs(bo), 99.5)
s3.imshow(bo, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
s3.plot(*e.boxy(fb["a_phase"], f["w"] / 2, 0, cx, cy, c=2), color=e.FOU, lw=1)
e.zoom(s3, cx, cy, 15 / ps)
s3.set_title("(f) imagen solo-barra ($m = 2, 4, 6$)", fontsize=6.5)
e.save(fig, "fourier")
