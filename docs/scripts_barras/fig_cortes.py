"""Figura: cortes de brillo por los ejes de la barra en la galaxia ejemplo, como en la literatura.
(a) imagen con la barra sobre x y las dos franjas de 5 px; (b) mu_r a lo largo del eje mayor y del menor,
mu(a) de las isofotas y un disco exponencial ajustado afuera; (c) contraste mu_menor - mu_mayor."""
import matplotlib.pyplot as plt
import numpy as np

import _estilo as e

HALF = 2                                 # franja de 2 * HALF + 1 = 5 px
DISK = (25, 42)                          # tramo del ajuste del disco [″]
DM_END = 0.25                            # contraste que se toma como fin de la barra [mag]

g = e.galaxy(e.EXAMPLE)
ps, t, b = g["ps"], g["t"], g["b"]
rot, cx, cy = e.on_x(g["img"], g["x0"], g["y0"], b["pa_bar"])
ic, jc = int(round(cy)), int(round(cx))


def mu(i):
    return 22.5 - 2.5 * np.log10(np.clip(i, 1e-5, None) / ps**2)


def folded(prof, c):
    """Promedio de los dos lados del centro, en función del radio [″]."""
    x = (np.arange(prof.size) - c) * ps
    return 0.5 * (np.interp(R, x, prof) + np.interp(R, -x[::-1], prof[::-1]))


R = np.arange(0, 45, ps)
major = folded(rot[ic - HALF:ic + HALF + 1, :].mean(0), cx)
minor = folded(rot[:, jc - HALF:jc + HALF + 1].mean(1), cy)
dm = mu(minor) - mu(major)
end = R[(R > 5) & (dm < DM_END)][0]

ok = np.isin(t["stop_code"], e.bar.GOOD_STOP)
mu_iso = mu(np.array(t["intens"])[ok])
fit = (g["a"] > DISK[0]) & (g["a"] < DISK[1])
k, c0 = np.polyfit(g["a"][fit], mu_iso[fit], 1)
print(f"h_disco = {1.0857 / k:.1f}″  a_eps = {b['a_eps']:.1f}″  fin del contraste (< {DM_END} mag) = {end:.1f}″  "
      f"contraste medio 7-13″ = {dm[(R > 7) & (R < 13)].mean():.2f} mag")

e.cm.style(e.W2)
fig = plt.figure(figsize=(e.W2, 0.36 * e.W2), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.1, 0.9])
a = fig.add_subplot(gs[0])
a.imshow(rot, origin="lower", cmap="gray_r", norm=e.asinh_norm(rot))
a.axhspan(ic - HALF - 0.5, ic + HALF + 0.5, color=e.ISO, alpha=0.4, lw=0)
a.axvspan(jc - HALF - 0.5, jc + HALF + 0.5, color=e.COMB, alpha=0.4, lw=0)
e.zoom(a, cx, cy, 30 / ps)
a.set_title("(a) barra sobre $x$ y las dos franjas", fontsize=6.5)

s = fig.add_subplot(gs[1])
s.plot(R, mu(major), color=e.ISO, lw=0.9, label="eje mayor de la barra")
s.plot(R, mu(minor), color=e.COMB, lw=0.9, ls="--", label="eje menor")
s.plot(g["a"], mu_iso, ".", ms=2, color=e.FOU, label=r"isofotas, $\mu(a)$")
s.plot(R, c0 + k * R, color=e.DEC, lw=0.8, label="disco exponencial")
s.set_ylim(26, 17.5)
s.set_ylabel(r"$\mu_r$ [mag arcsec$^{-2}$]")
s.legend(loc="upper right", frameon=False, fontsize=5.5)
s.set_title("(b) cortes por el centro", fontsize=6.5)

c = fig.add_subplot(gs[2])
c.plot(R, dm, color="0.2", lw=0.9)
c.axhline(0, color="0.6", lw=0.5)
c.axhline(DM_END, color="0.6", lw=0.5, ls=":")
c.set_ylim(-1.2, 1.6)
c.set_ylabel(r"$\mu_{\rm menor} - \mu_{\rm mayor}$ [mag]")
c.set_title("(c) contraste barra / interbarra", fontsize=6.5)

for ax in (s, c):
    ax.axvline(b["a_eps"], color=e.ISO, lw=0.8)
    ax.axvline(end, color="0.4", lw=0.8, ls=":")
    ax.axvline(e.A_REF_EXAMPLE, color=e.REF, lw=0.7, ls="--")
    ax.set_xlim(0, 42 if ax is s else 30)
    ax.set_xlabel(r"radio [$^{\prime\prime}$]")
e.save(fig, "cortes")
