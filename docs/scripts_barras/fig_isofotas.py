"""Figura: ajuste de isofotas en la galaxia ejemplo. (a) imagen con elipses ajustadas y la de la barra;
(b) elipticidad, (c) ángulo de posición y (d) brillo superficial en función del semieje mayor, con
a_ε (máximo de ε), a_up (fin de la barra) y el semilargo de Valeria."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

import _estilo as e

g = e.galaxy(e.EXAMPLE)
ps, t, b = g["ps"], g["t"], g["b"]
ok = np.isin(t["stop_code"], e.bar.GOOD_STOP)
mu = 22.5 - 2.5 * np.log10(np.clip(np.array(t["intens"])[ok], 1e-6, None) / ps**2)
print(f"a_eps = {b['a_eps']:.2f}″  eps_bar = {b['eps_bar']:.3f}  a_up = {b['a_up']:.2f}″  "
      f"PA_bar (imagen) = {b['pa_bar']:.1f}°  b = {b['b_bar']:.2f}″  isofotas buenas: {ok.sum()}/{len(t)}")

e.cm.style(e.W2)
fig = plt.figure(figsize=(e.W2, 0.42 * e.W2), layout="constrained")
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.3])
a = fig.add_subplot(gs[:, 0])
a.imshow(g["img"], origin="lower", cmap="gray_r", norm=e.asinh_norm(g["img"], 99.9, 0.02))
for r in t[ok][::5]:
    a.add_patch(Ellipse((r["x0"], r["y0"]), 2 * r["sma"], 2 * r["sma"] * (1 - r["ellipticity"]),
                        angle=float(r["pa"]), fill=False, lw=0.4, color="0.55"))
a.add_patch(Ellipse((g["x0"], g["y0"]), 2 * b["a_eps"] / ps, 2 * b["b_bar"] / ps, angle=b["pa_bar"],
                    fill=False, lw=1.2, color=e.ISO))
e.zoom(a, g["x0"], g["y0"], 35 / ps)
a.set_title("(a) elipses ajustadas; azul: la de $a_\\varepsilon$", fontsize=6.5)

panels = [(g["eps"], r"$\varepsilon$", "(b)"), (g["pa"], r"PA [$^\circ$]", "(c)"), (mu, r"$\mu_r$ [mag arcsec$^{-2}$]", "(d)")]
for k, (y, lab, tag) in enumerate(panels):
    s = fig.add_subplot(gs[k, 1])
    s.plot(g["a"], y, ".", ms=2, color=e.ISO)
    s.axvspan(0, e.A_MIN, color="0.9", lw=0)
    s.axvline(b["a_eps"], color=e.ISO, lw=0.8)
    s.axvline(b["a_up"], color=e.ISO, lw=0.8, ls=":")
    s.axvline(e.A_REF_EXAMPLE, color=e.REF, lw=0.7, ls="--")
    s.set_xlim(0, 45)
    s.set_ylabel(lab)
    s.text(0.97, 0.08 if k == 0 else 0.8, tag, transform=s.transAxes, fontsize=6.5, ha="right")
    if k < 2:
        s.tick_params(labelbottom=False)
    else:
        s.invert_yaxis()
        s.set_xlabel(r"semieje mayor $a$ [$^{\prime\prime}$]")
e.save(fig, "isofotas")
