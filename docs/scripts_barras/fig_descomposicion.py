"""Figura: descomposición 2D (bulbo + disco + barra Ferrers + anillo, pyimfit) de la galaxia con el
problema de ángulo, 39627884146461249. (a) imagen con las barras de cada método; (b) modelo;
(c) residuo (imagen - modelo)/σ; (d) imagen - (bulbo + disco + anillo): queda solo la barra.
Ajuste guardado en data_barras/fit_884_ring.json (parámetros) y model_884_ring.npy (modelo)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyimfit
from astropy.io import fits
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import _estilo as e

TID, A_REF = 39627884146461249, 10.88
P = json.load(open(e.DATA / "fit_884_ring.json"))["P"]
model = np.load(e.DATA / "model_884_ring.npy")
g = e.galaxy(TID)
ps, img = g["ps"], g["img"]
ivar = fits.getdata(e.CUT / f"{TID}_desi_r_ivar.fits", 1)
psf = fits.getdata(e.CUT / f"{TID}_desi_r_psf.fits").astype(float)
cfg = [f"X0 {P['X0']}", f"Y0 {P['Y0']}", "FUNCTION FerrersBar2D",
       f"PA {P['barPA']} fixed", f"ell {P['barell']} fixed", f"c0 {P['c0']} fixed", "n 2 fixed",
       f"I_0 {P['barI0']} fixed", f"a_bar {P['abar']} fixed"]
bar_img = pyimfit.Imfit(pyimfit.parse_config(cfg), psf=psf).getModelImage(shape=img.shape)
row = pd.read_parquet(e.DATA / "bars_common_desi_v4.parquet").set_index("TARGETID").loc[TID]
x0, y0 = P["X0"] - 1, P["Y0"] - 1
A, Bm = P["abar"], P["abar"] * (1 - P["barell"])
print(f"2D: a_bar = {A * ps:.1f}″ b = {Bm * ps:.1f}″ PA = {P['barPA']:.1f}° c0 = {P['c0']:.2f} "
      f"anillo R = {P['R'] * ps:.1f}″ | isofotas a = {row.a_eps:.1f}″ PA = {row.pa_bar:.1f}° (PA_eps {row.pa_eps:.1f}°) "
      f"| Fourier a = {row.a_fourier:.1f}″ PA = {row.pa_fourier:.1f}° | Valeria a = {A_REF}″")

e.cm.style(e.W2)
fig, ax = plt.subplots(1, 4, figsize=(e.W2, 0.29 * e.W2), layout="constrained")
norm = e.asinh_norm(img)
ax[0].imshow(img, origin="lower", cmap="gray_r", norm=norm)
ax[0].plot(*e.boxy(A, Bm, P["barPA"] + 90, x0, y0, c=P["c0"] + 2), color=e.DEC, lw=1)
ax[0].plot(*e.boxy(row.a_eps / ps, row.W_comb / 2 / ps, row.pa_eps + 90, x0, y0), color=e.COMB, lw=1)
ax[0].plot(*e.boxy(row.a_fourier / ps, row.W_fourier / 2 / ps, row.pa_fourier + 90, x0, y0), color=e.FOU, lw=0.8)
ax[0].add_patch(Circle((x0, y0), A_REF / ps, fill=False, color=e.REF, ls="--", lw=0.6))
ax[0].legend([Line2D([], [], color=c) for c in (e.COMB, e.FOU, e.DEC)], ["combinada", "Fourier", "2D"],
             fontsize=5, frameon=False, loc="lower left")
ax[1].imshow(model, origin="lower", cmap="gray_r", norm=norm)
ax[2].imshow((img - model) * np.sqrt(np.clip(ivar, 0, None)), origin="lower", cmap="RdBu_r", vmin=-10, vmax=10)
ax[3].imshow(img - model + bar_img, origin="lower", cmap="gray_r", norm=norm)
ax[3].plot(*e.boxy(A, Bm, P["barPA"] + 90, x0, y0, c=P["c0"] + 2), color=e.DEC, lw=0.8)
for a_, t in zip(ax, ["(a) barras", "(b) modelo", r"(c) residuo$/\sigma$", "(d) imagen $-$ (bulbo+disco+anillo)"]):
    e.zoom(a_, x0, y0, 28 / ps)
    a_.set_title(t, fontsize=6.5)
e.save(fig, "descomposicion")
