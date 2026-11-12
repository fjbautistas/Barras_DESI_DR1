"""
Bar width from the Fourier modes of the light (Ohta et al. 1990; Aguerri et al. 2000).

On each circle of radius r the light is written as
    I(r, φ) = A0(r) + Σ_m [a_m(r) cos mφ + b_m(r) sin mφ].
A0 is the axisymmetric light (bulge + disk); the bar lives in the even modes
m = 2, 4, 6. The image of those modes alone has no bulge by construction:
the bar width is measured there.
"""
import numpy as np
from scipy.ndimage import map_coordinates, rotate

M_MAX = 6


def polar_modes(img, mask, x0, y0, rmax, n_phi=360, m_max=M_MAX):
    """A0 and (a_m, b_m) for m = 1..m_max on circles r = 1..rmax px.

    Each circle is sampled at n_phi angles; masked samples are dropped and the
    coefficients are a least-squares fit, so a masked star does not bias them.
    φ is counted from +x, counter-clockwise (the photutils convention).
    """
    data = np.where(mask, np.nan, img)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    m = np.arange(1, m_max + 1)
    design = np.column_stack([np.ones_like(phi), np.cos(np.outer(phi, m)), np.sin(np.outer(phi, m))])
    r = np.arange(1, int(rmax) + 1, dtype=float)
    coef = np.full((len(r), 1 + 2 * m_max), np.nan)
    for i, ri in enumerate(r):
        vals = map_coordinates(data, [y0 + ri * np.sin(phi), x0 + ri * np.cos(phi)], order=1, cval=np.nan)
        ok = np.isfinite(vals)
        if ok.sum() > 0.8 * n_phi:
            coef[i] = np.linalg.lstsq(design[ok], vals[ok], rcond=None)[0]
    return r, coef[:, 0], coef[:, 1:1 + m_max], coef[:, 1 + m_max:]


def bar_only_image(shape, x0, y0, r, a, b, r_max, modes=(2, 4, 6)):
    """Image of the even modes inside r_max: the light that is not axisymmetric."""
    yy, xx = np.indices(shape)
    rr, pp = np.hypot(xx - x0, yy - y0), np.arctan2(yy - y0, xx - x0)
    out = np.zeros(shape)
    inside = rr <= r_max
    for m in modes:
        am = np.interp(rr[inside], r, np.nan_to_num(a[:, m - 1]))
        bm = np.interp(rr[inside], r, np.nan_to_num(b[:, m - 1]))
        out[inside] += am * np.cos(m * pp[inside]) + bm * np.sin(m * pp[inside])
    return out


def _fwhm(prof, i_axis):
    """Full width at half maximum [px] of the peak next to i_axis, contiguous pixels only."""
    ip = i_axis - 3 + np.argmax(prof[i_axis - 3:i_axis + 4])
    half = prof[ip] / 2
    lo, hi = ip, ip
    while lo > 0 and prof[lo - 1] >= half:
        lo -= 1
    while hi < len(prof) - 1 and prof[hi + 1] >= half:
        hi += 1
    return hi - lo + 1


def bar_width(shape, x0, y0, r, a, b, a_bar, phi_bar, at=(0.5, 0.75), half_width=2):
    """Bar width W [px]: FWHM across the bar-only image, mean over cuts at ±at × a_bar.

    The image is rotated so the bar (phi_bar, deg from +x) lies on x. Near the
    centre the m ≥ 2 modes vanish by construction, so the cuts are taken at
    half and three quarters of the bar length; each averages 2 half_width + 1 columns.
    """
    bo = bar_only_image(shape, x0, y0, r, a, b, a_bar)
    c = (np.array(shape) - 1) / 2
    shifted = np.roll(bo, (int(round(c[0] - y0)), int(round(c[1] - x0))), axis=(0, 1))
    rot = rotate(shifted, phi_bar, reshape=False, order=1)
    i_axis = int(round(c[0]))
    widths = [_fwhm(rot[:, int(round(c[1] + s * f * a_bar)) - half_width:
                        int(round(c[1] + s * f * a_bar)) + half_width + 1].mean(1), i_axis)
              for f in at for s in (-1, 1)]
    return float(np.mean(widths))
