"""
Bar length and width of one galaxy.

    measure_fits("galaxy_r.fits", z=0.04)      # a FITS image of one band
    measure_image(img, pixscale=0.262, z=0.04)  # the same, from an array
    measure_galaxy(row, "outputs/cutouts")      # downloads the Legacy cutout first

Steps: sky and masks → isophotes (ellipse fit) → bar length and PA from the
first peak of ε (src/bar.py) → bar width from the bulge-free Fourier image
(src/fourier.py) → physical size with z.

Output, projected on the sky (as in Alonso et al. 2024):
    a_bar   bar semi-length [arcsec]           L_kpc = 2 a_bar  [kpc]
    W_bar   bar full width  [arcsec]           W_kpc            [kpc]
    pa_bar  bar PA [deg, north through east]   eps_bar  ellipticity of the bar isophote
    b_iso   semi-minor axis of that isophote [arcsec]; it includes bulge light,
            like contour widths such as the reference method's
    flag_eps, flag_end, flag_pa   quality flags (src/bar.py)
The galaxy must be at the centre of the image, which must be sky-dominated
at its edges.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from src import bar, cutout, fourier, isophotes

A_MIN = 2.0  # arcsec: the bar is searched beyond ~1.5 PSF FWHM (DR10 r); provisional


def to_astro_pa(pa_img):
    """photutils angle (from +x, CCW) to PA from north through east (north +y, east -x)."""
    return (pa_img - 90) % 180


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
    b = bar.bar_from_profile(a, eps, pa, a_min)

    r, _, am, bm = fourier.polar_modes(img, mask, x0, y0, rmax=r_gal)
    w = fourier.bar_width(img.shape, x0, y0, r, am, bm, b["a_eps"] / pixscale, b["pa_bar"]) * pixscale

    scale = bar.kpc_per_arcsec(z) if z is not None else np.nan
    return {
        "a_bar": b["a_eps"], "W_bar": w, "pa_bar": to_astro_pa(b["pa_bar"]), "eps_bar": b["eps_bar"],
        "b_iso": b["a_eps"] * (1 - b["eps_bar"]),
        "L_kpc": 2 * b["a_eps"] * scale, "W_kpc": w * scale,
        "flag_eps": b["flag_eps"], "flag_end": b["flag_end"], "flag_pa": b["flag_pa"],
        "x0": x0, "y0": y0, "sky": sky, "sky_std": sky_std, "a_max_iso": a.max(),
    }


def measure_fits(path, z=None, pixscale=None, hdu=0, **kw):
    """Bar from a FITS image; the pixel scale is read from the WCS if not given."""
    img, header = fits.getdata(path, hdu, header=True)
    if pixscale is None:
        pixscale = float(np.sqrt(abs(np.linalg.det(WCS(header).pixel_scale_matrix))) * 3600)
    return measure_image(img, pixscale, z, **kw)


def measure_galaxy(row, cut_dir, survey="desi", a_min=A_MIN):
    """Bar of one row of a sample (TARGETID, RA, DEC, Z, SHAPE_R): Legacy cutout, then measure."""
    name = str(row.TARGETID)
    size = int(cutout.cutout_size(row.SHAPE_R, survey))
    paths = cutout.download(row.RA, row.DEC, size, cut_dir, name, survey)
    out = measure_image(fits.getdata(paths["fits"]), cutout.pixscale(survey), row.Z, a_min,
                        iso_path=cut_dir / f"{name}_{survey}_iso.ecsv")
    return {"TARGETID": row.TARGETID, "Z": row.Z, "survey": survey, **out}
