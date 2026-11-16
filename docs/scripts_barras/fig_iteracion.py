"""Figura del anexo: ajuste de una isofota en la galaxia ejemplo, a = 5″.
(a) elipse de partida y elipse ajustada; (b) brillo a lo largo de la de partida con sus armónicos 1 y 2;
(c) lo mismo para la ajustada. Mismo muestreo que src/isophotes.py (mediana por sector, centro fijo)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from photutils.isophote import EllipseGeometry, EllipseSample
from photutils.isophote.fitter import EllipseFitter
from photutils.isophote.harmonics import first_and_second_harmonic_function, fit_first_and_second_harmonics

import _estilo as e

A_ARCSEC, EPS0, PA0 = 5.0, 0.30, 140.0   # radio y geometría de partida (PA en grados, desde +x)

g = e.galaxy(e.EXAMPLE)
ps = g["ps"]
data = np.ma.masked_array(g["img"], g["mask"])
sma = A_ARCSEC / ps


def sample(eps, pa):
    geom = EllipseGeometry(g["x0"], g["y0"], sma, eps, pa, fix_center=True)
    s = EllipseSample(data, sma, geometry=geom, integrmode="median", sclip=3.0, nclip=3)
    s.update(geom.fix)
    return s


iso = EllipseFitter(sample(EPS0, np.radians(PA0))).fit(maxgerr=2.0)
eps1, pa1 = iso.sample.geometry.eps, np.degrees(iso.sample.geometry.pa)
print(f"partida: eps = {EPS0}, PA = {PA0}°  ->  eps = {eps1:.3f}, PA = {pa1:.1f}°, "
      f"{iso.niter} iteraciones, stop_code = {iso.stop_code}")

e.cm.style(e.W2)
fig = plt.figure(figsize=(e.W2, 0.34 * e.W2), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 1, 1])
a = fig.add_subplot(gs[0])
a.imshow(g["img"], origin="lower", cmap="gray_r", norm=e.asinh_norm(g["img"], 99.9, 0.02))
a.add_patch(Ellipse((g["x0"], g["y0"]), 2 * sma, 2 * sma * (1 - EPS0), angle=PA0,
                    fill=False, lw=1.2, color=e.FOU, ls="--"))
a.add_patch(Ellipse((g["x0"], g["y0"]), 2 * sma, 2 * sma * (1 - eps1), angle=pa1, fill=False, lw=1.2, color=e.ISO))
e.zoom(a, g["x0"], g["y0"], 9 / ps)
a.set_title("(a) partida y ajustada", fontsize=6.5)

axes = []
for k, (eps, pa, color, tag) in enumerate([(EPS0, PA0, e.FOU, "(b) partida"), (eps1, pa1, e.ISO, "(c) ajustada")]):
    phi, _, inten = sample(eps, np.radians(pa)).extract()
    c = fit_first_and_second_harmonics(phi, inten)[0]
    o = np.argsort(phi)
    s = fig.add_subplot(gs[k + 1], sharey=axes[0] if axes else None)
    s.plot(np.degrees(phi), inten, ".", ms=2, color="0.55")
    s.plot(np.degrees(phi[o]), first_and_second_harmonic_function(phi[o], c), color=color, lw=1.2)
    s.axhline(c[0], color=e.REF, lw=0.6)
    s.set_xlim(0, 360), s.set_xticks([0, 90, 180, 270, 360])
    s.set_xlabel(r"ángulo sobre la elipse $\varphi$ [$^\circ$]")
    s.set_title(f"{tag}: $\\varepsilon$ = {eps:.2f}, PA = {pa:.0f}°\n"
                f"$A_1$ = {c[1]:.3f}  $B_1$ = {c[2]:.3f}  $A_2$ = {c[3]:.3f}  $B_2$ = {c[4]:.3f}", fontsize=6)
    print(tag, "I0 = %.3f  A1 B1 A2 B2 =" % c[0], np.round(c[1:], 3))
    axes.append(s)
axes[0].set_ylabel("brillo [nmgy/px]")
axes[1].tick_params(labelleft=False)
e.save(fig, "iteracion")
