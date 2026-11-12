"""Figura: tres errores del primer código y su corrección, cada uno en una galaxia real.
(a) centro en una estrella: el píxel más brillante de todo el segmento (antes) frente al más brillante
cerca del centro del recorte (ahora);
(b) isofotas congeladas: fit_image de photutils frente al ajuste radio por radio;
(c) isofotas "pegadas": ajuste que parte de ε = 0.2, PA = 0 y acepta código 2 sin moverse, frente al
actual (inicio en los momentos de la luz y código 2 sin moverse descartado)."""
import matplotlib.pyplot as plt
import numpy as np
from photutils.isophote import Ellipse, EllipseGeometry, EllipseSample
from photutils.isophote.fitter import EllipseFitter
from scipy.ndimage import center_of_mass

import _estilo as e

STAR, FROZEN, STUCK = 39632986676660740, 39633240629182912, 39628022831121141


def old_centre(img, gal, box=5):
    y, x = np.unravel_index(np.argmax(np.where(gal, img, -np.inf)), img.shape)
    cut = img[y - box:y + box + 1, x - box:x + box + 1]
    dy, dx = center_of_mass(np.clip(cut, 0, None))
    return x - box + dx, y - box + dy


def eps_profile(t, ps):
    ok = np.isin(np.array(t["stop_code"]), (0, 2))
    return np.array(t["sma"])[ok] * ps, np.array(t["ellipticity"])[ok]


e.cm.style(e.W2)
fig, ax = plt.subplots(1, 3, figsize=(e.W2, 0.33 * e.W2), layout="constrained")

g = e.galaxy(STAR)
ps = g["ps"]
xo, yo = old_centre(g["img"], g["gal"])
print(f"(a) {STAR}: centro viejo a {np.hypot(xo - g['x0'], yo - g['y0']) * ps:.1f}″ del nuevo")
ax[0].imshow(g["img"], origin="lower", cmap="gray_r", norm=e.asinh_norm(g["img"]))
ax[0].plot(xo, yo, "x", color=e.FOU, ms=6, mew=1.2, label="antes")
ax[0].plot(g["x0"], g["y0"], "+", color=e.ISO, ms=7, mew=1.2, label="ahora")
e.zoom(ax[0], g["x0"], g["y0"], 45 / ps)
ax[0].legend(fontsize=5.5, frameon=False, loc="lower left")
ax[0].set_title("(a) centro en una estrella", fontsize=6.5)

g = e.galaxy(FROZEN)
ps = g["ps"]
data = np.ma.masked_array(g["img"], g["mask"])
yy, xx = np.nonzero(g["gal"])
rmax = np.hypot(xx - g["x0"], yy - g["y0"]).max()
old = Ellipse(data, EllipseGeometry(g["x0"], g["y0"], 15, 0.2, 0.0)).fit_image(
    sma0=15, minsma=1, maxsma=rmax, step=0.05, fix_center=True, sclip=3, nclip=3,
    integrmode="median", maxgerr=2.0).to_table()
a_old, e_old = eps_profile(old, ps)
print(f"(b) {FROZEN}: fit_image llega a {a_old.max():.1f}″; radio por radio a {g['a'].max():.1f}″")
ax[1].plot(g["a"], g["eps"], ".", ms=2, color=e.ISO, label="radio por radio")
ax[1].plot(a_old, e_old, "o", ms=2.5, mfc="none", mec=e.FOU, mew=0.6, label="photutils fit\\_image")
ax[1].axvline(e.A_MIN, color="0.7", lw=0.6)
ax[1].set_title("(b) isofotas congeladas", fontsize=6.5)

g = e.galaxy(STUCK)
ps = g["ps"]
data = np.ma.masked_array(g["img"], g["mask"])
eps0, pa0, raw = 0.2, 0.0, []
for sma in 15 * 1.05 ** np.arange(0, 30):
    geom = EllipseGeometry(g["x0"], g["y0"], sma, eps0, pa0, fix_center=True)
    s = EllipseSample(data, sma, geometry=geom, sclip=3.0, nclip=3, integrmode="median")
    s.update(geom.fix)
    iso = EllipseFitter(s).fit(maxgerr=2.0)
    raw.append((sma * ps, iso.sample.geometry.eps, iso.stop_code))
    if iso.stop_code in (0, 2):
        eps0, pa0 = iso.sample.geometry.eps, iso.sample.geometry.pa
raw = np.array(raw)
good = np.isin(raw[:, 2], (0, 2))
stuck = good & (np.abs(raw[:, 1] - 0.2) < 1e-9)
print(f"(c) {STUCK}: {stuck.sum()} isofotas con ε = 0.2 exacto entre {raw[stuck, 0].min():.1f} y "
      f"{raw[stuck, 0].max():.1f}″; a_ε actual = {g['b']['a_eps']:.1f}″")
ax[2].plot(g["a"], g["eps"], ".", ms=2, color=e.ISO, label="ahora")
ax[2].plot(raw[good, 0], raw[good, 1], "o", ms=2.5, mfc="none", mec=e.FOU, mew=0.6, label="antes (desde 3.9$''$)")
ax[2].plot(raw[stuck, 0], raw[stuck, 1], "s", ms=3, mfc="none", mec="k", mew=0.6, label=r"$\varepsilon = 0.2$ sin moverse")
ax[2].set_title("(c) isofotas pegadas al inicio", fontsize=6.5)
for a_ in ax[1:]:
    a_.set_xlim(0, 15)
    a_.set_ylim(0, 1)
    a_.set_xlabel(r"semieje mayor $a$ [$^{\prime\prime}$]")
    a_.set_ylabel(r"$\varepsilon$")
    a_.legend(fontsize=5, frameon=False, loc="upper left")
e.save(fig, "problemas")
