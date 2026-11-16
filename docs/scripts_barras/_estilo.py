"""Estilo, rutas y utilidades comunes de las figuras de barras.tex (usa Paper_plots/common.py).

Colores de los métodos (orden azul -> morado -> naranja -> verde):
isofotas azul, barra combinada morado, Fourier naranja, descomposición 2D verde;
la referencia (Valeria, Alonso et al. 2024) en negro a trazos.
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, "/Users/frank/Documents/PhD/PhD_tesis/semestre-2026-II/Tesis_III/Paper/Paper_plots")
import common as cm                     # noqa: E402

# common.py deja importado el paquete src de Galaxy_pairs: se descarta y se usa el de este proyecto
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]
sys.path.insert(0, str(ROOT))

from checks import bar_full as bar         # noqa: E402  (a_up, b_bar: versiones completas, no están en src)
from checks import fourier_full as fourier  # noqa: E402
from src import isophotes                 # noqa: E402
from src.cutout import cutout_paths, pixscale  # noqa: E402
from src.measure import A_MIN            # noqa: E402
from tools.compare_reference import stats  # noqa: E402
from tools.steps import bar_on_x as on_x, boxy  # noqa: E402,F401

OUT = HERE.parent / "figures_barras"
DATA = HERE.parent / "data_barras"
CUT = ROOT / "outputs" / "cutouts"
W1 = cm.COL_W                           # 3.5 in, una columna
W2 = 6.3                                # ~\textwidth del documento (A4, márgenes 2.5 cm)
ISO, COMB, FOU, DEC = cm.SERIES         # azul, morado, naranja, verde
REF = "k"
EXAMPLE = 39627835815496281             # galaxia ejemplo
A_REF_EXAMPLE = 14.05                   # semilargo de Valeria para el ejemplo [″]


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("->", OUT / f"{name}.pdf")


def galaxy(tid, survey="desi"):
    """Imagen sin cielo, máscara, centro, tabla de isofotas, perfiles y barra por isofotas."""
    ps = pixscale(survey)
    img = fits.getdata(cutout_paths(CUT, str(tid), survey)["fits"]).astype(float)
    sky, _, mask, gal = isophotes.sky_and_mask(img)
    img = img - sky
    x0, y0 = isophotes.galaxy_centre(img, gal)
    t = Table.read(CUT / f"{tid}_{survey}_iso.ecsv")
    a, eps, pa = bar.profiles(t, ps)
    b = bar.bar_from_profile(a, eps, pa, A_MIN)
    return dict(img=img, mask=mask, gal=gal, x0=x0, y0=y0, t=t, a=a, eps=eps, pa=pa, b=b, ps=ps)


def fourier_of(g, rmax_arcsec=40.0):
    """Modos, largo y ancho de Fourier, y ancho combinado (en px) de una galaxia de galaxy()."""
    ps = g["ps"]
    r, a0, am, bm = fourier.polar_modes(g["img"], g["mask"], g["x0"], g["y0"], rmax=rmax_arcsec / ps)
    fb = fourier.bar_length(r, a0, am, bm, r_min=A_MIN / ps)
    w = fourier.bar_width(g["img"].shape, g["x0"], g["y0"], r, am, bm, fb["a_phase"], fb["phi2_bar"])
    wc = fourier.bar_width(g["img"].shape, g["x0"], g["y0"], r, am, bm, g["b"]["a_eps"] / ps, g["b"]["pa_bar"])
    return dict(r=r, a0=a0, am=am, bm=bm, fb=fb, w=w, wc=wc)


def asinh_norm(img, pct=99.8, soft=0.03):
    from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
    return ImageNormalize(img, interval=PercentileInterval(pct), stretch=AsinhStretch(soft))


def zoom(ax, x0, y0, half_px):
    ax.set_xlim(x0 - half_px, x0 + half_px), ax.set_ylim(y0 - half_px, y0 + half_px)
    ax.set_xticks([]), ax.set_yticks([])


def ratio_stats(x, ref):
    """N, mediana del cociente, semiancho 16-84 % y fracción a menos del 20 %."""
    r = stats(np.asarray(x, float), np.asarray(ref, float))
    return r["N"], r["median"], r["half_68"], r["within_20"]
