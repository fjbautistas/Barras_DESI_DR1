"""Figura del anexo: definiciones. (a) una isofota: a, b, ε, PA, θ y r(φ); (b) las cantidades de la barra,
con la barra sobre x: a_bar = a_ε, L, W, b_iso y la isofota de a_up. Esquema, sin datos."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse

import _estilo as e

A, B, TH = 3.0, 1.4, 30.0                # isofota de ejemplo: semiejes y θ [deg]
t = np.radians(TH)
u = np.array([np.cos(t), np.sin(t)])     # eje mayor
v = np.array([-np.sin(t), np.cos(t)])    # eje menor
arrow = dict(arrowstyle="->", lw=1.0, shrinkA=0, shrinkB=0)

e.cm.style(e.W2)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(e.W2, 0.45 * e.W2), layout="constrained")

ax.add_patch(Ellipse((0, 0), 2 * A, 2 * B, angle=TH, fill=False, lw=1.2, color=e.ISO))
ax.plot([-A * u[0], A * u[0]], [-A * u[1], A * u[1]], color=e.COMB, lw=0.5, ls=":")
ax.plot([-3.8, 3.8], [0, 0], color="0.6", lw=0.5, ls=":")
ax.plot(0, 0, "k+", ms=5)
ax.text(-1.45, -0.5, r"$(x_0, y_0)$")
ax.annotate("", A * u, (0, 0), arrowprops=dict(arrow, color=e.COMB))
ax.text(*(0.55 * A * u + 0.3 * v), r"$a$", color=e.COMB, fontsize=8)
ax.annotate("", B * v, (0, 0), arrowprops=dict(arrow, color=e.FOU))
ax.text(*(0.6 * B * v - 0.35 * u), r"$b$", color=e.FOU, fontsize=8)
ax.add_patch(Arc((0, 0), 2.2, 2.2, theta1=0, theta2=TH, color="k", lw=0.7))
ax.text(1.25, 0.2, r"$\theta$", fontsize=8)
ax.annotate("", (0, 3.3), (0, 0), arrowprops=dict(arrow, color="k"))
ax.text(0.1, 3.25, "N")
ax.annotate("", (-3.4, 0), (0, 0), arrowprops=dict(arrow, color="k"))
ax.text(-3.6, 0.15, "E")
ax.add_patch(Arc((0, 0), 4.0, 4.0, theta1=90, theta2=90 + TH + 90, color=e.DEC, lw=1.0))
ax.text(-2.2, 1.3, "PA", color=e.DEC, fontsize=8)
phi = np.radians(-65)
r = A * B / np.hypot(B * np.cos(phi), A * np.sin(phi))
d = np.array([np.cos(t + phi), np.sin(t + phi)])
ax.plot([0, r * d[0]], [0, r * d[1]], color="0.3", lw=0.7, ls="--")
ax.text(*(r * d + [0.12, -0.1]), r"$r(\varphi)$", color="0.3")
ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=TH - 65, theta2=TH, color="0.3", lw=0.6))
ax.text(0.78, -0.42, r"$\varphi$", color="0.3")
ax.set_xlim(-4, 4), ax.set_ylim(-3.4, 3.6), ax.set_aspect("equal"), ax.axis("off")
ax.set_title("(a) una isofota", fontsize=6.5)

for s, eps in zip(np.linspace(0.4, 3.0, 7), np.linspace(0.15, 0.65, 7)):
    bx.add_patch(Ellipse((0, 0), 2 * s, 2 * s * (1 - eps), fill=False, lw=0.4, color="0.6"))
bx.add_patch(Ellipse((0, 0), 6.0, 6.0 * 0.35, fill=False, lw=1.2, color=e.ISO))
bx.add_patch(Ellipse((0, 0), 7.4, 7.4 * 0.79, angle=-35, fill=False, lw=1.0, color=e.FOU, ls="--"))
bx.plot(*e.boxy(3.0, 0.55, 0, 0, 0), color=e.COMB, lw=1.2)
bx.annotate("", (3.0, -2.2), (0, -2.2), arrowprops=dict(arrowstyle="<->", lw=0.8, color=e.ISO))
bx.text(1.5, -2.55, r"$a_{\rm bar} = a_\varepsilon$", color=e.ISO, ha="center")
bx.annotate("", (-3.0, -3.0), (3.0, -3.0), arrowprops=dict(arrowstyle="<->", lw=0.8, color="k"))
bx.text(0, -3.35, r"$L = 2\,a_{\rm bar}$", ha="center")
bx.annotate("", (1.6, 0.55), (1.6, -0.55), arrowprops=dict(arrowstyle="<->", lw=0.8, color=e.COMB))
bx.text(1.72, -0.12, r"$W$", color=e.COMB)
bx.annotate("", (-0.5, 1.05), (-0.5, 0), arrowprops=dict(arrowstyle="<->", lw=0.6, color=e.ISO))
bx.text(-1.25, 1.2, r"$b_{\rm iso}$", color=e.ISO)
bx.text(-3.9, 3.35, r"isofota de $a_{\rm up}$", color=e.FOU)
bx.set_xlim(-4, 4), bx.set_ylim(-3.6, 3.7), bx.set_aspect("equal"), bx.axis("off")
bx.set_title("(b) la barra, girada al eje $x$", fontsize=6.5)
e.save(fig, "definiciones")
