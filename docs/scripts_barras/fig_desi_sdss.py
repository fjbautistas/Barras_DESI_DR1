"""Figura: la misma galaxia ejemplo medida con el mismo código sobre la imagen de DESI Legacy
(0.262″/px) y la de SDSS (0.396″/px): ε(a) y φ2(r), y las barras. Separa el efecto de los datos
del efecto del método."""
import matplotlib.pyplot as plt
import numpy as np

import _estilo as e

res = {}
for s in ("desi", "sdss"):
    g = e.galaxy(e.EXAMPLE, s)
    f = e.fourier_of(g)
    res[s] = (g, f)
    print(f"{s}: a_eps = {g['b']['a_eps']:.2f}″  b = {g['b']['b_bar']:.2f}″  eps = {g['b']['eps_bar']:.2f}  "
          f"a_fase = {f['fb']['a_phase'] * g['ps']:.2f}″  W_Fourier = {f['w'] * g['ps']:.2f}″  "
          f"W_comb = {f['wc'] * g['ps']:.2f}″  isofotas hasta {g['a'].max():.1f}″")

e.cm.style(e.W2)
fig = plt.figure(figsize=(e.W2, 0.33 * e.W2), layout="constrained")
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.3, 1.3])
for k, (s, col) in enumerate([("desi", e.ISO), ("sdss", e.COMB)]):
    g, f = res[s]
    ps = g["ps"]
    rot, cx, cy = e.on_x(g["img"], g["x0"], g["y0"], g["b"]["pa_bar"])
    a = fig.add_subplot(gs[:, k])
    a.imshow(rot, origin="lower", cmap="gray_r", norm=e.asinh_norm(rot))
    a.plot(*e.boxy(g["b"]["a_eps"] / ps, f["wc"] / 2 / ps, 0, cx, cy), color=e.COMB, lw=1)
    a.plot(*e.boxy(f["fb"]["a_phase"], f["w"] / 2, f["fb"]["phi2_bar"] - g["b"]["pa_bar"], cx, cy),
           color=e.FOU, lw=0.8)
    e.zoom(a, cx, cy, 22 / ps)
    a.set_title(f"({'ab'[k]}) {s.upper()}", fontsize=6.5)
s1 = fig.add_subplot(gs[:, 2])
s2 = fig.add_subplot(gs[:, 3])
for s, col, mk in [("desi", e.ISO, "."), ("sdss", e.COMB, "x")]:
    g, f = res[s]
    s1.plot(g["a"], g["eps"], mk, ms=2, color=col, label=s.upper())
    s2.plot(f["r"] * g["ps"], f["fb"]["phi2"], mk, ms=1.5, color=col, label=s.upper())
s1.set_xlim(0, 30), s2.set_xlim(0, 30)
s1.set_xlabel(r"$a$ [$^{\prime\prime}$]"), s1.set_ylabel(r"$\varepsilon$")
s2.set_xlabel(r"$r$ [$^{\prime\prime}$]"), s2.set_ylabel(r"$\varphi_2$ [$^\circ$]")
s1.legend(fontsize=5.5, frameon=False), s1.set_title("(c) elipticidad", fontsize=6.5)
s2.set_title("(d) fase de $m = 2$", fontsize=6.5)
e.save(fig, "desi_sdss")
