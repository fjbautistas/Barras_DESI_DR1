"""
2D decomposition bulge + disk + bar with imfit (Erwin 2015; pyimfit), the
open counterpart of GALFIT.

Model, convolved with the Legacy coadd PSF and weighted by the inverse
variance of each pixel:
  bulge  Sersic       (PA, ell, n, I_e, r_e)
  disk   Exponential  (PA, ell, I_0, h)
  bar    FerrersBar2D (PA, ell, c0, n, I_0, a_bar): a generalised ellipse
         |x/a|^(c0+2) + |y/b|^(c0+2) = 1 with a Ferrers profile
         I = I_0 (1 - (m/a_bar)^2)^n inside it; a_bar is where it reaches 0
  ring   GaussianRing (PA, ell, A, R_ring, sigma_r): without it the bar
         absorbs the light of inner rings and grows.
The bar comes out as a component of its own: semi-major axis a_bar,
semi-minor axis a_bar (1 - ell), PA, all from one fit.
Starting values come from the isophotes (bar a, ε, PA; disk ε, PA).
imfit PA is counter-clockwise from +y, i.e. the astronomical PA in
Legacy cutouts (north up, east left).
"""
from pathlib import Path

import numpy as np
import pyimfit
from astropy.io import fits

from src.cutout import SURVEYS, VIEWER, _fetch


def download_psf_ivar(ra, dec, size, out_dir, name, survey="desi", band="r"):
    """Coadd PSF at the position and inverse-variance cutout of the same field."""
    layer, scale = SURVEYS[survey]
    out_dir = Path(out_dir)
    psf = out_dir / f"{name}_{survey}_{band}_psf.fits"
    ivar = out_dir / f"{name}_{survey}_{band}_ivar.fits"
    if not psf.exists():
        _fetch(f"{VIEWER}/coadd-psf/?ra={ra:.6f}&dec={dec:.6f}&layer={layer}&bands={band}", psf)
    if not ivar.exists():
        _fetch(f"{VIEWER}/fits-cutout?ra={ra:.6f}&dec={dec:.6f}&layer={layer}"
               f"&pixscale={scale}&size={size}&bands={band}&invvar", ivar)
    return fits.getdata(psf).astype(float), fits.getdata(ivar, 1).astype(float)


COMPONENTS = [  # (name, imfit function, parameters in imfit order)
    ("bulge", "Sersic", ["PA", "ell", "n", "I_e", "r_e"]),
    ("disk", "Exponential", ["PA", "ell", "I_0", "h"]),
    ("bar", "FerrersBar2D", ["PA", "ell", "c0", "n", "I_0", "a_bar"]),
    ("ring", "GaussianRing", ["PA", "ell", "A", "R_ring", "sigma_r"]),
]


def model_config(x0, y0, iso, img, scale_px):
    """imfit configuration with starting values and limits from the isophotes.

    The same four components for every galaxy, so nothing is added by hand:
    a ring the galaxy does not have just fits a negligible amplitude.
    iso: dict with a_eps [px], eps_bar, pa_bar, eps_disk, pa_disk (astronomical
    PA, deg) and h0 [px], a first guess of the disk scale length.
    """
    yy, xx = np.indices(img.shape)
    rr = np.hypot(xx - x0, yy - y0)

    def at(r):  # median intensity on the circle of radius r [px]
        return float(np.median(img[np.abs(rr - r) < 1.5]))

    a, h = iso["a_eps"], iso["h0"]
    i_disk0 = max(at(2 * a) * np.exp(2 * a / h), 1e-4)
    i_bar0 = max(at(a / 2) - i_disk0 * np.exp(-a / 2 / h), 1e-4)
    i_ring = max(0.2 * at(1.3 * a), 1e-4)
    re0 = 1.5 / scale_px
    i_bulge = max(at(re0), 1e-4)
    pb, pd = iso["pa_bar"], iso["pa_disk"]
    ed = min(iso["eps_disk"], 0.8)
    return f"""
X0   {x0 + 1:.2f}   {x0:.2f},{x0 + 2:.2f}
Y0   {y0 + 1:.2f}   {y0:.2f},{y0 + 2:.2f}
FUNCTION Sersic
PA    {pd:.1f}   0,360
ell   0.1        0,0.6
n     2.0        0.5,6
I_e   {i_bulge:.4g}   {i_bulge / 100:.4g},{i_bulge * 100:.4g}
r_e   {re0:.2f}   1,{a:.2f}
FUNCTION Exponential
PA    {pd:.1f}   {pd - 30:.1f},{pd + 30:.1f}
ell   {ed:.2f}   0,0.85
I_0   {i_disk0:.4g}   {i_disk0 / 100:.4g},{i_disk0 * 100:.4g}
h     {h:.2f}   {h / 4:.2f},{h * 4:.2f}
FUNCTION FerrersBar2D
PA    {pb:.1f}   {pb - 30:.1f},{pb + 30:.1f}
ell   {min(iso['eps_bar'], 0.9):.2f}   0.2,0.95
c0    0.5        -0.5,3
n     2          fixed
I_0   {i_bar0:.4g}   {i_bar0 / 100:.4g},{i_bar0 * 100:.4g}
a_bar {a:.2f}   {a / 3:.2f},{a * 3:.2f}
FUNCTION GaussianRing
PA    {pd:.1f}   {pd - 30:.1f},{pd + 30:.1f}
ell   {ed:.2f}   0,0.85
A     {i_ring:.4g}   {i_ring / 1000:.4g},{i_ring * 100:.4g}
R_ring {1.3 * a:.2f}   {0.8 * a:.2f},{4 * a:.2f}
sigma_r {0.3 * a:.2f}   1,{2 * a:.2f}
"""


def fit(img, ivar, mask, psf, config):
    """Levenberg-Marquardt fit; returns the parameters by component, the
    reduced χ², whether it converged and the model image."""
    fitter = pyimfit.Imfit(pyimfit.parse_config(config.splitlines()), psf=psf)
    bad = mask | ~np.isfinite(img) | (ivar <= 0)
    fitter.loadData(np.where(bad, 0, img), mask=bad.astype(float),
                    error=np.where(bad, 1, ivar), error_type="weight")  # weight = 1/σ² = invvar
    result = fitter.doFit()
    names = ["X0", "Y0"] + [f"{c}:{q}" for c, _, ps in COMPONENTS for q in ps]
    params = dict(zip(names, fitter.getRawParameters()))
    return params, result.fitStatReduced, result.fitConverged, fitter.getModelImage()


def component_image(params, name, psf, shape):
    """Model image of one fitted component (PSF-convolved)."""
    func, ps = next((f, q) for c, f, q in COMPONENTS if c == name)
    cfg = [f"X0 {params['X0']}", f"Y0 {params['Y0']}", f"FUNCTION {func}"]
    cfg += [f"{q} {params[f'{name}:{q}']} fixed" for q in ps]
    return pyimfit.Imfit(pyimfit.parse_config(cfg), psf=psf).getModelImage(shape=shape)
