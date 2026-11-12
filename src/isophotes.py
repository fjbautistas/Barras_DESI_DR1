"""
Steps 3 and 4: sky, masks and isophote fit of the central galaxy.

Angles are kept as photutils returns them: radians from the +x axis,
counter-clockwise (east is -x, north +y in Legacy cutouts).
"""
import numpy as np
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from photutils.isophote import EllipseGeometry, EllipseSample, IsophoteList
from photutils.isophote.fitter import EllipseFitter
from photutils.segmentation import detect_sources, make_2dgaussian_kernel
from scipy.ndimage import binary_dilation, center_of_mass

GOOD_STOP = (0, 2)  # converged, or max iterations reached
STUCK = 6  # own code: max iterations reached and the geometry never left its start


def sky_and_mask(img, nsigma=2.0, npixels=10, dilate=5):
    """Sky level and noise, mask of every source except the central galaxy.

    Sources are detected nsigma above the sky on a smoothed image; the
    central galaxy is the segment under the image centre. Other sources are
    dilated by `dilate` px to cover their faint wings. Stars projected on the
    galaxy belong to its segment and are left to the sigma clipping of the fit.
    """
    _, med, std = sigma_clipped_stats(img, sigma=3.0)
    smooth = convolve(img - med, make_2dgaussian_kernel(3.0, size=5))
    segm = detect_sources(smooth, nsigma * std, npixels)

    cy, cx = (np.array(img.shape) - 1) // 2
    galaxy = segm.data == segm.data[cy, cx]
    if segm.data[cy, cx] == 0:
        raise ValueError("no source at the image centre")
    others = binary_dilation((segm.data > 0) & ~galaxy, iterations=dilate)

    _, sky, sky_std = sigma_clipped_stats(img, mask=binary_dilation(segm.data > 0, iterations=dilate))
    return sky, sky_std, others & ~galaxy, galaxy


def galaxy_centre(img, galaxy, box=5, r_search=10):
    """Light-weighted centre in a (2 box + 1) px box around the brightest galaxy
    pixel within r_search px of the image centre.

    The cutout is centred on the DESI position, so the nucleus is there; a
    bright star touching the galaxy belongs to its segment and would win a
    search over the whole segment.
    """
    yy, xx = np.indices(img.shape)
    c = (np.array(img.shape) - 1) / 2
    near = galaxy & (np.hypot(xx - c[1], yy - c[0]) <= r_search)
    y, x = np.unravel_index(np.argmax(np.where(near, img, -np.inf)), img.shape)
    cut = img[y - box:y + box + 1, x - box:x + box + 1]
    dy, dx = center_of_mass(np.clip(cut, 0, None))
    return x - box + dx, y - box + dy


def light_moments(img, mask, x0, y0, radius):
    """Ellipticity and orientation [rad, from +x CCW] of the light within
    `radius` px, from its second moments: the starting ellipse of the fit."""
    yy, xx = np.indices(img.shape)
    dx, dy = xx - x0, yy - y0
    w = np.where(~mask & (np.hypot(dx, dy) <= radius), np.clip(img, 0, None), 0)
    mxx, myy, mxy = (w * dx * dx).sum(), (w * dy * dy).sum(), (w * dx * dy).sum()
    lam = np.linalg.eigvalsh([[mxx, mxy], [mxy, myy]])
    eps = 1 - np.sqrt(lam[0] / lam[1])
    return float(np.clip(eps, 0.05, 0.6)), float(0.5 * np.arctan2(2 * mxy, mxx - myy))


def fit_isophotes(img, mask, x0, y0, maxsma, sma0=15.0, step=0.05, maxgerr=2.0):
    """Ellipse fit from 1 px to maxsma, geometric step, fixed centre.

    Each radius is fitted on its own, starting from the last converged
    ellipse. photutils' fit_image freezes the geometry of every larger
    radius once two consecutive isophotes have a flat or rising profile,
    which bars and rings produce: one bad isophote then ruins the rest.

    The intensity along each ellipse is the median (robust to HII regions,
    dust lanes and stars); maxgerr = 2 instead of 0.5 for the shallow
    gradient of the bar region.

    Returns the photutils table: sma [px], intens, ellipticity, pa [deg],
    their errors and stop_code (0 = converged, 2 = max iterations,
    STUCK = max iterations without moving from the starting geometry).
    """
    data = np.ma.masked_array(img, mask)

    def run(smas, eps, pa, going_inwards):
        out = []
        for sma in smas:
            geom = EllipseGeometry(x0, y0, sma, eps, pa, fix_center=True)
            sample = EllipseSample(data, sma, geometry=geom, sclip=3.0, nclip=3, integrmode="median")
            sample.update(geom.fix)
            iso = EllipseFitter(sample).fit(maxgerr=maxgerr, going_inwards=going_inwards)
            # code 2 can return its starting ellipticity untouched: not a fit
            if iso.stop_code == 2 and iso.sample.geometry.eps == eps:
                iso.stop_code = STUCK
            out.append(iso)
            if iso.stop_code in GOOD_STOP:
                eps, pa = iso.sample.geometry.eps, iso.sample.geometry.pa
        return out

    n_out = int(np.log(maxsma / sma0) / np.log1p(step)) + 1
    n_in = int(np.log(sma0) / np.log1p(step))
    outward = run(sma0 * (1 + step) ** np.arange(n_out), *light_moments(img, mask, x0, y0, sma0), False)
    converged = [i for i in outward if i.stop_code == 0]
    start = ((converged[0].sample.geometry.eps, converged[0].sample.geometry.pa) if converged
             else light_moments(img, mask, x0, y0, sma0))
    inward = run(sma0 / (1 + step) ** np.arange(1, n_in + 1), *start, True)

    isos = outward + inward
    if not any(i.stop_code in GOOD_STOP for i in isos):
        raise RuntimeError("isophote fit failed")
    return IsophoteList(sorted(isos, key=lambda i: i.sma)).to_table()
