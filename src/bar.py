"""
The bar read from the isophote profiles ε(a), PA(a).

Bar (Marinova & Jogee 2007): the first local maximum of ε beyond a_min with
ε ≥ 0.25 that is followed by a drop of at least 0.1. Its semi-major axis
a_eps is the bar semi-length (Erwin 2005, lower limit); its PA is the bar
orientation.

Angles in degrees, in the image frame of photutils (from +x, counter-clockwise).
"""
import numpy as np
from astropy.cosmology import Planck18
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from src.isophotes import GOOD_STOP


def axis_diff(x, y):
    """Difference between two axis orientations [deg], in [0, 90]."""
    return np.abs((np.asarray(x) - y + 90) % 180 - 90)


def profiles(t, pixscale):
    """a [arcsec], ε and PA [deg, 0–180] of the converged isophotes."""
    ok = np.isin(np.array(t["stop_code"]), GOOD_STOP)
    return (np.array(t["sma"])[ok] * pixscale, np.array(t["ellipticity"])[ok],
            np.array(t["pa"])[ok] % 180)


def bar_peak(a, eps, a_min, deps=0.1, eps_bar_min=0.25, smooth=3):
    """Index of the bar in ε(a). An inclined or lopsided outer disk can be
    more elongated than the bar, so the global maximum is not the bar; it is
    only the fallback when no peak meets the criteria (the flags say so)."""
    inner = np.flatnonzero(a >= a_min)
    eps_s = median_filter(eps, smooth, mode="nearest")
    peaks, _ = find_peaks(eps_s)
    for i in peaks[(peaks >= inner[0]) & (eps_s[peaks] >= eps_bar_min)]:
        if (eps_s[i + 1:] < eps_s[i] - deps).any():
            return i
    return inner[np.argmax(eps[inner])]


def bar_from_profile(a, eps, pa, a_min, deps=0.1, dpa=10.0, eps_bar_min=0.25):
    """Bar semi-length a_eps [arcsec], ellipticity, PA and quality flags:
    flag_eps  ε_bar ≥ eps_bar_min
    flag_end  ε drops by deps or PA turns by dpa beyond a_eps (the bar ends)
    flag_pa   PA constant within dpa over a_eps/2 ≤ a ≤ a_eps
    """
    i = bar_peak(a, eps, a_min, deps, eps_bar_min)
    eps_bar, pa_bar, a_eps = eps[i], pa[i], a[i]
    plateau = (a >= max(a_min, a_eps / 2)) & (a <= a_eps)
    return {
        "a_eps": a_eps, "eps_bar": eps_bar, "pa_bar": pa_bar,
        "flag_eps": bool(eps_bar >= eps_bar_min),
        "flag_end": bool(((eps[i + 1:] < eps_bar - deps) | (axis_diff(pa[i + 1:], pa_bar) > dpa)).any()),
        "flag_pa": bool(axis_diff(pa[plateau], pa_bar).max() <= dpa),
    }


def kpc_per_arcsec(z):
    """Proper scale at redshift z, Planck18."""
    return Planck18.kpc_proper_per_arcmin(z).value / 60
