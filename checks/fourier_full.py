"""
Fourier analysis of the bar (Ohta et al. 1990; Aguerri et al. 2000).

On each circle of radius r the light is written as
    I(r, φ) = A0(r) + Σ_m [a_m(r) cos mφ + b_m(r) sin mφ].
A0 is the axisymmetric light (bulge + disk); the bar lives in the even
modes m = 2, 4, 6, with constant phase φ2 along it.

Bar length, two published criteria, both fixed beforehand:
- intensity ratio (Aguerri et al. 2000): bar I_b = A0 + I2 + I4 + I6 and
  interbar I_ib = A0 - I2 + I4 - I6; the bar ends where I_b/I_ib falls below
  half way between its maximum and its minimum.
- phase: the bar ends where φ2 departs by more than dphi from its value at
  the peak of A2/A0.
Width: from the bar-only image I - A0 (even modes), where the axisymmetric
bulge is removed by construction.
"""
import numpy as np
from scipy.ndimage import map_coordinates, median_filter, rotate
from scipy.signal import find_peaks

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


def amplitude_phase(a, b, m):
    """Amplitude I_m and phase φ_m [deg, in 0..180/m] of mode m."""
    amp = np.hypot(a[:, m - 1], b[:, m - 1])
    return amp, np.rad2deg(np.arctan2(b[:, m - 1], a[:, m - 1]) / m) % (360 / m)


def _first_peak(y, start, min_value):
    """Index of the first local maximum of y at or after start with y ≥ min_value."""
    peaks, _ = find_peaks(np.nan_to_num(y, nan=-np.inf))
    peaks = peaks[(peaks >= start) & (y[peaks] >= min_value)]
    return peaks[0] if len(peaks) else None


def bar_length(r, a0, a, b, r_min, dphi=10.0, a2_min=0.2, smooth=5):
    """Bar semi-length [px] by the phase and intensity-ratio criteria.

    Spiral arms and rings also carry m = 2, often more than the bar, so the
    bar is the FIRST peak of A2/A0 beyond r_min with A2/A0 ≥ a2_min, not the
    global maximum. Profiles are smoothed by a running median of `smooth` px.
    """
    i2, phi2 = amplitude_phase(a, b, 2)
    i4, _ = amplitude_phase(a, b, 4)
    i6, _ = amplitude_phase(a, b, 6)
    a2 = median_filter(i2 / a0, smooth, mode="nearest")
    ratio = median_filter((a0 + i2 + i4 + i6) / (a0 - i2 + i4 - i6), smooth, mode="nearest")
    start = np.searchsorted(r, r_min)
    out = {"a2": a2, "phi2": phi2, "ratio": ratio, "a_phase": np.nan, "a_ratio": np.nan,
           "a2_max": np.nan, "r_a2_max": np.nan, "phi2_bar": np.nan, "ratio_level": np.nan}

    j = _first_peak(a2, start, a2_min)
    if j is None:
        return out
    out.update(a2_max=a2[j], r_a2_max=r[j], phi2_bar=phi2[j])
    off = (np.arange(len(r)) > j) & (np.abs((phi2 - phi2[j] + 90) % 180 - 90) > dphi)
    if off.any():
        out["a_phase"] = r[np.flatnonzero(off)[0]]

    # ratio: its first peak, and half way down to the next local minimum
    k = _first_peak(ratio, start, 1.0)
    if k is not None:
        mins, _ = find_peaks(-np.nan_to_num(ratio[k:], nan=np.inf))
        low = ratio[k + mins[0]] if len(mins) else np.nanmin(ratio[k:])
        level = 0.5 * (ratio[k] + low)
        below = (np.arange(len(r)) > k) & (ratio < level)
        out["ratio_level"] = level
        if below.any():
            out["a_ratio"] = r[np.flatnonzero(below)[0]]
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

    Near the centre the m ≥ 2 modes vanish by construction, so the cuts are
    taken at half and three quarters of the bar length; each cut averages
    2 half_width + 1 columns.
    """
    bo = bar_only_image(shape, x0, y0, r, a, b, a_bar)
    c = (np.array(shape) - 1) / 2
    shifted = np.roll(bo, (int(round(c[0] - y0)), int(round(c[1] - x0))), axis=(0, 1))
    rot = rotate(shifted, phi_bar, reshape=False, order=1)
    i_axis = int(round(c[0]))
    widths = []
    for f in at:
        for s in (-1, 1):
            xc = int(round(c[1] + s * f * a_bar))
            widths.append(_fwhm(rot[:, xc - half_width:xc + half_width + 1].mean(1), i_axis))
    return float(np.mean(widths))


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
