"""
Bar length and width of one galaxy.

    measure_fits("galaxy_r.fits", z=0.04)      # a FITS image of one band
    measure_jpg("galaxy.jpg", 0.262, z=0.04)    # last resort: the bar comes out short
    measure_image(img, pixscale=0.262, z=0.04)  # the same, from an array
    measure_galaxy(row, "outputs/cutouts")      # downloads the Legacy cutout first

Steps: sky and masks → isophotes (ellipse fit) → bar length and PA from the
first peak of ε beyond max(A_MIN, R_MIN_KPC) (src/bar.py) → bar width from the
bulge-free Fourier image (src/fourier.py) → disc from the outer isophotes and
deprojection → physical size with z (NaN without z).

Output, projected on the sky (as in Alonso et al. 2024):
    a_bar   bar semi-length [arcsec]           L_kpc = 2 a_bar  [kpc]
    W_bar   bar full width  [arcsec]           W_kpc            [kpc]
    pa_bar  bar PA [deg, north through east]   eps_bar  ellipticity of the bar isophote
    b_iso   semi-minor axis of that isophote [arcsec]; it includes bulge light,
            like contour widths such as the reference method's
    flag_eps, flag_end, flag_pa   quality flags (src/bar.py)
and the same sizes on the disc plane (*_dep), with the disc ε and PA and the
angle dphi between bar and disc. The disc window in μ_r assumes nanomaggies.
The galaxy must be at the centre of the image, which must be sky-dominated
at its edges.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from src import bar, cutout, fourier, isophotes

A_MIN = 2.0  # arcsec: the bar is searched beyond ~1.5 PSF FWHM (DR10 r); provisional
R_MIN_KPC = 1.0  # kpc: and beyond nuclear rings/bars, resolved in nearby galaxies (S4G test)


def to_astro_pa(pa_img):
    """photutils angle (from +x, CCW) to PA from north through east (north +y, east -x)."""
    return (pa_img - 90) % 180


def disc(t, pixscale, a_bar, mu=(23.5, 25.0)):
    """Disc ε and PA (image frame): isophotes beyond 1.5 a_bar with mu[0] < μ_r < mu[1]
    (else the last 3); median ε, PA averaged on the doubled angle."""
    t = t[np.isin(np.array(t["stop_code"]), isophotes.GOOD_STOP)]
    m_mu = 22.5 - 2.5 * np.log10(np.clip(np.array(t["intens"]), 1e-9, None) / pixscale ** 2)
    out = np.flatnonzero(np.array(t["sma"]) * pixscale > 1.5 * a_bar)
    sel = out[(m_mu[out] > mu[0]) & (m_mu[out] < mu[1])]
    sel = sel if sel.size >= 3 else out[-3:]
    if not sel.size:
        return np.nan, np.nan
    t2 = np.radians(2 * np.array(t["pa"])[sel])
    return np.median(np.array(t["ellipticity"])[sel]), np.degrees(np.arctan2(np.sin(t2).mean(), np.cos(t2).mean())) / 2 % 180


def deproject(length, dphi, eps_disc):
    """Length at dphi [deg] from the disc major axis, on the disc plane (thin disc, cos i = 1 - ε)."""
    return length * np.hypot(np.cos(np.radians(dphi)), np.sin(np.radians(dphi)) / (1 - eps_disc))


def measure_image(img, pixscale, z=None, a_min=A_MIN, iso_path=None):
    """Bar of the galaxy at the centre of img (2D array, one band, linear flux)."""
    img = np.asarray(img, dtype=float)
    sky, sky_std, mask, galaxy = isophotes.sky_and_mask(img)
    img = img - sky
    x0, y0 = isophotes.galaxy_centre(img, galaxy)
    yy, xx = np.nonzero(galaxy)
    r_gal = np.hypot(xx - x0, yy - y0).max()

    t = isophotes.fit_isophotes(img, mask, x0, y0, maxsma=r_gal)
    if iso_path:
        t.write(iso_path, overwrite=True)
    a, eps, pa = bar.profiles(t, pixscale)
    scale = bar.kpc_per_arcsec(z) if z is not None else np.nan
    a_min = max(a_min, R_MIN_KPC / scale) if z is not None else a_min
    b = bar.bar_from_profile(a, eps, pa, a_min)

    r, _, am, bm = fourier.polar_modes(img, mask, x0, y0, rmax=r_gal)
    w = fourier.bar_width(x0, y0, r, am, bm, b["a_eps"] / pixscale, b["pa_bar"]) * pixscale

    eps_d, pa_d = disc(t, pixscale, b["a_eps"])
    dphi = bar.axis_diff(b["pa_bar"], pa_d)
    a_dep, w_dep = deproject(b["a_eps"], dphi, eps_d), deproject(w, 90 - dphi, eps_d)
    return {
        "a_bar": b["a_eps"], "W_bar": w, "pa_bar": to_astro_pa(b["pa_bar"]), "eps_bar": b["eps_bar"],
        "b_iso": b["a_eps"] * (1 - b["eps_bar"]),
        "L_kpc": 2 * b["a_eps"] * scale, "W_kpc": w * scale,
        "a_bar_dep": a_dep, "W_bar_dep": w_dep, "L_kpc_dep": 2 * a_dep * scale, "W_kpc_dep": w_dep * scale,
        "eps_disc": eps_d, "pa_disc": to_astro_pa(pa_d), "dphi": dphi, "a_min": a_min,
        "flag_eps": b["flag_eps"], "flag_end": b["flag_end"], "flag_pa": b["flag_pa"],
        "x0": x0, "y0": y0, "sky": sky, "sky_std": sky_std, "a_max_iso": a.max(),
    }


def measure_fits(path, z=None, pixscale=None, hdu=0, **kw):
    """Bar from a FITS image; the pixel scale is read from the WCS if not given."""
    img, header = fits.getdata(path, hdu, header=True)
    if pixscale is None:
        pixscale = float(np.sqrt(abs(np.linalg.det(WCS(header).pixel_scale_matrix))) * 3600)
    return measure_image(img, pixscale, z, **kw)


def measure_jpg(path, pixscale, z=None, **kw):
    """Bar from a colour JPG (green channel ≈ r). The 8-bit non-linear stretch saturates the centre:
    in the 448 reference bars a_bar / reference has median 0.74, and W and the disc are not reliable."""
    img = np.flipud(np.asarray(Image.open(path).convert("RGB"))[:, :, 1]).astype(float)
    return measure_image(img, pixscale, z, **kw)


def measure_galaxy(row, cut_dir, survey="desi", a_min=A_MIN):
    """Bar of one row of a sample (TARGETID, RA, DEC, Z, SHAPE_R; Z may be NaN): Legacy cutout, then measure."""
    name, z = str(row.TARGETID), (row.Z if np.isfinite(row.Z) else None)
    size = int(cutout.cutout_size(row.SHAPE_R, survey))
    paths = cutout.download(row.RA, row.DEC, size, cut_dir, name, survey)
    out = measure_image(fits.getdata(paths["fits"]), cutout.pixscale(survey), z, a_min,
                        iso_path=cut_dir / f"{name}_{survey}_iso.ecsv")
    return {"TARGETID": row.TARGETID, "Z": row.Z, "survey": survey, **out}
