"""
Steps 2 to 6 for one galaxy: cutout, sky and masks, isophotes, bar, deprojection.

Output lengths: semi-axes a (along the bar) and b (across it) in arcsec,
projected and deprojected, plus L = 2a and W = 2b in kpc (deprojected).
PAs are astronomical (north through east) on output.
"""
import numpy as np
from astropy.io import fits

from checks import bar_full as bar, fourier_full as fourier
from src import cutout, isophotes

A_MIN = 2.0  # arcsec, ~1.5 PSF FWHM in DR10 r; provisional


def to_astro_pa(pa_img):
    """photutils angle (from +x, CCW) to PA from north through east (north +y, east -x)."""
    return (pa_img - 90) % 180


def measure_galaxy(row, cut_dir, survey="desi", a_min=A_MIN):
    name = str(row.TARGETID)
    scale_px = cutout.pixscale(survey)
    paths = cutout.download(row.RA, row.DEC, int(cutout.cutout_size(row.SHAPE_R, survey)),
                            cut_dir, name, survey)
    img = fits.getdata(paths["fits"]).astype(float)

    sky, sky_std, mask, galaxy = isophotes.sky_and_mask(img)
    img = img - sky
    x0, y0 = isophotes.galaxy_centre(img, galaxy)
    yy, xx = np.nonzero(galaxy)
    t = isophotes.fit_isophotes(img, mask, x0, y0, maxsma=np.hypot(xx - x0, yy - y0).max())
    t.write(cut_dir / f"{name}_{survey}_iso.ecsv", overwrite=True)

    r, a0, am, bm = fourier.polar_modes(img, mask, x0, y0, rmax=np.hypot(xx - x0, yy - y0).max())
    fb = fourier.bar_length(r, a0, am, bm, r_min=a_min / scale_px)
    w_px = (fourier.bar_width(img.shape, x0, y0, r, am, bm, fb["a_phase"], fb["phi2_bar"])
            if np.isfinite(fb["a_phase"]) else np.nan)

    a, eps, pa = bar.profiles(t, scale_px)
    b = bar.bar_from_profile(a, eps, pa, a_min)
    # combined bar: length and PA from the isophotes, width from the bar-only
    # (bulge-free) Fourier image, cut at 0.5 and 0.75 of that length
    w_comb_px = fourier.bar_width(img.shape, x0, y0, r, am, bm, b["a_eps"] / scale_px, b["pa_bar"])
    eps_disk, pa_disk, n_disk = bar.disk_orientation(a, eps, pa)
    a_dep, b_dep = bar.deproject(b["a_eps"], b["b_bar"], b["pa_bar"], eps_disk, pa_disk)
    a_up_dep = bar.deproject_length(b["a_up"], b["pa_bar"], eps_disk, pa_disk)
    scale = bar.kpc_per_arcsec(row.Z)

    return {
        "TARGETID": row.TARGETID, "survey": survey, "Z": row.Z, "sky": sky, "sky_std": sky_std,
        "n_iso": len(a), "a_max_iso": a.max(),
        "a_eps": b["a_eps"], "a_up": b["a_up"], "b_bar": b["b_bar"], "eps_bar": b["eps_bar"],
        "pa_bar": to_astro_pa(b["pa_bar"]), "pa_eps": to_astro_pa(b["pa_eps"]),
        "eps_disk": eps_disk, "pa_disk": to_astro_pa(pa_disk), "n_iso_disk": n_disk,
        "a_eps_dep": a_dep, "b_bar_dep": b_dep, "a_up_dep": a_up_dep,
        "kpc_per_arcsec": scale,
        "L_kpc": 2 * a_dep * scale, "L_up_kpc": 2 * a_up_dep * scale, "W_kpc": 2 * b_dep * scale,
        "flag_eps": b["flag_eps"], "flag_end": b["flag_end"], "flag_pa": b["flag_pa"],
        "flag_inclined": eps_disk > 0.5,
        "x0": x0, "y0": y0,
        # Fourier (projected): semi-length from the φ2 phase, full width W from the bar-only image
        "a_fourier": fb["a_phase"] * scale_px, "W_fourier": w_px * scale_px,
        "a2_max": fb["a2_max"], "r_a2_max": fb["r_a2_max"] * scale_px,
        "pa_fourier": to_astro_pa(fb["phi2_bar"]),
        "flag_fourier": bool(np.isfinite(fb["a_phase"])),
        "W_comb": w_comb_px * scale_px,
        # physical size, projected (the deprojected L_kpc, W_kpc above wait for a reliable disk)
        "L_proj_kpc": 2 * b["a_eps"] * scale, "W_proj_kpc": w_comb_px * scale_px * scale,
    }
