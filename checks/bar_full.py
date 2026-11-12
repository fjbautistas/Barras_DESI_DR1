"""
Steps 5 and 6: the bar read from the isophote profiles, and its deprojection.

Bar criteria (Marinova & Jogee 2007): ε rises to a maximum ε_bar ≥ 0.25 over
a region of nearly constant PA, then drops at the end of the bar.
Length (Erwin 2005): a_eps = a(ε_max) is the lower limit; the upper limit
a_up is the first radius beyond it where ε has dropped by deps or the PA
has turned by more than dpa. Width: b_bar = a_eps (1 - ε_bar).

Angles in degrees, in the image frame of photutils (from +x, counter-clockwise).
"""
import numpy as np
from astropy.cosmology import Planck18
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

GOOD_STOP = (0, 2)  # converged, or max iterations reached


def axis_diff(x, y):
    """Difference between two axis orientations [deg], in [0, 90]."""
    return np.abs((np.asarray(x) - y + 90) % 180 - 90)


def profiles(t, pixscale):
    ok = np.isin(np.array(t["stop_code"]), GOOD_STOP)
    return (np.array(t["sma"])[ok] * pixscale, np.array(t["ellipticity"])[ok],
            np.array(t["pa"])[ok] % 180)


def bar_peak(a, eps, a_min, deps=0.1, eps_bar_min=0.25, smooth=3):
    """Index of the bar in ε(a): the FIRST local maximum beyond a_min with
    ε ≥ eps_bar_min that is followed by a drop of at least deps.

    An inclined or lopsided outer disk can be more elongated than the bar, so
    the global maximum is not the bar. If no peak meets the criteria, the
    global maximum is returned and the flags will say so.
    """
    inner = np.flatnonzero(a >= a_min)
    eps_s = median_filter(eps, smooth, mode="nearest")
    peaks, _ = find_peaks(eps_s)
    for i in peaks[(peaks >= inner[0]) & (eps_s[peaks] >= eps_bar_min)]:
        if (eps_s[i + 1:] < eps_s[i] - deps).any():
            return i
    return inner[np.argmax(eps[inner])]


def bar_from_profile(a, eps, pa, a_min, deps=0.1, dpa=10.0, eps_bar_min=0.25):
    """Bar parameters from ε(a), PA(a) (a in arcsec, a > a_min only)."""
    inner = a >= a_min
    i = bar_peak(a, eps, a_min, deps, eps_bar_min)
    eps_bar, a_eps = eps[i], a[i]

    # PA of the bar: mean over the bar, a_eps/2 ≤ a ≤ a_eps, not the single
    # isophote at a_eps, which may already twist towards the arms or the disk
    plateau = inner & (a >= a_eps / 2) & (a <= a_eps)
    pa_bar = axis_mean(pa[plateau])

    after = np.arange(len(a)) > i
    end = after & ((eps < eps_bar - deps) | (axis_diff(pa, pa_bar) > dpa))
    a_up = a[np.flatnonzero(end)[0]] if end.any() else np.nan

    return {
        "a_eps": a_eps, "a_up": a_up, "eps_bar": eps_bar, "pa_bar": pa_bar, "pa_eps": pa[i],
        "b_bar": a_eps * (1 - eps_bar),
        "flag_eps": bool(eps_bar >= eps_bar_min),
        "flag_end": bool(np.isfinite(a_up)),
        "flag_pa": bool(axis_diff(pa[plateau], pa_bar).max() <= dpa),
    }


def axis_mean(pa):
    """Mean of axis orientations [deg]: doubled-angle circular mean, since 0° and 180° are the same axis."""
    ang = np.deg2rad(2 * np.asarray(pa))
    return float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2 % 180)


def disk_orientation(a, eps, pa, frac=0.7):
    """ε and PA of the disk: outer isophotes, a ≥ frac × the last one."""
    outer = a >= frac * a.max()
    return float(np.median(eps[outer])), axis_mean(pa[outer]), int(outer.sum())


def deproject(a, b, pa, eps_disk, pa_disk):
    """Semi-axes of the bar ellipse in the plane of a thin disk.

    The sky ellipse (a, b, pa) is written as a quadratic form in the frame of
    the disk major axis; stretching the minor-axis direction by 1/cos i maps
    it to the disk plane, where its eigenvalues give 1/a'² and 1/b'².
    """
    cos_i = 1 - eps_disk
    phi = np.deg2rad(pa - pa_disk)
    rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    q_sky = rot @ np.diag([1 / a**2, 1 / b**2]) @ rot.T
    s_inv = np.diag([1.0, cos_i])
    lam = np.linalg.eigvalsh(s_inv @ q_sky @ s_inv)
    return 1 / np.sqrt(lam[0]), 1 / np.sqrt(lam[1])


def deproject_length(a, pa, eps_disk, pa_disk):
    """One radius along the bar axis, taken to the disk plane."""
    phi = np.deg2rad(pa - pa_disk)
    return a * np.sqrt(np.cos(phi)**2 + np.sin(phi)**2 / (1 - eps_disk)**2)


def kpc_per_arcsec(z):
    return Planck18.kpc_proper_per_arcmin(z).value / 60
