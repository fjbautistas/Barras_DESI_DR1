"""
Figures of each step of the method (notebooks/Barra_ejemplo.ipynb), and the
two drawing helpers shared with plot_gallery.py: boxy (the drawn bar) and
bar_on_x (the image rotated with the bar on x). No measuring here.
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from matplotlib.patches import Ellipse
from scipy.ndimage import rotate

CMAP = "gray_r"


def _show(ax, img, x0, y0, half, cmap=CMAP):
    norm = ImageNormalize(img, interval=PercentileInterval(99.8), stretch=AsinhStretch(0.03))
    ax.imshow(img, origin="lower", cmap=cmap, norm=norm)
    ax.set_xlim(x0 - half, x0 + half), ax.set_ylim(y0 - half, y0 + half)
    ax.set_xticks([]), ax.set_yticks([])


def boxy(a, b, angle_deg, x0, y0, c=3.0, n=300):
    """Outline |x/a|^c + |y/b|^c = 1 rotated by angle_deg (from +x, CCW): the drawn bar."""
    t = np.linspace(0, 2 * np.pi, n)
    x = a * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / c)
    y = b * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / c)
    p = np.deg2rad(angle_deg)
    return x0 + x * np.cos(p) - y * np.sin(p), y0 + x * np.sin(p) + y * np.cos(p)


def bar_on_x(img, x0, y0, angle_deg):
    """Image moved to the centre (x0, y0) and rotated so that angle_deg (from +x) lies on +x.

    ndimage.rotate turns counter-clockwise in (row, col), i.e. clockwise as
    shown with origin="lower", so +angle_deg brings that direction to +x.
    """
    c = (np.array(img.shape) - 1) / 2
    shifted = np.roll(img, (int(round(c[0] - y0)), int(round(c[1] - x0))), axis=(0, 1))
    return rotate(shifted, angle_deg, reshape=False, order=1), c[1], c[0]


def show_image(img, jpg=None):
    fig, ax = plt.subplots(1, 2 if jpg is not None else 1, figsize=(9, 4.5), squeeze=False)
    c = (np.array(img.shape) - 1) / 2
    _show(ax[0, 0], img, c[1], c[0], c[0])
    ax[0, 0].set_title("FITS, band r (what is measured)")
    if jpg is not None:
        ax[0, 1].imshow(jpg), ax[0, 1].set_xticks([]), ax[0, 1].set_yticks([])
        ax[0, 1].set_title("colour JPG g, r, z (only to look at)")
    fig.tight_layout()


def show_mask(img, mask, galaxy):
    fig, ax = plt.subplots(figsize=(5, 5))
    c = (np.array(img.shape) - 1) / 2
    _show(ax, img, c[1], c[0], c[0])
    over = np.zeros(img.shape + (4,))
    over[galaxy] = [0.12, 0.47, 0.71, 0.35]
    over[mask] = [1.0, 0.5, 0.05, 0.6]
    ax.imshow(over, origin="lower")
    ax.set_title("blue: the galaxy   orange: masked")


def show_isophotes(img, t, x0, y0, half, every=4):
    fig, ax = plt.subplots(figsize=(5, 5))
    _show(ax, img, x0, y0, half)
    sma, eps, pa = (np.array(t[c])[::every] for c in ("sma", "ellipticity", "pa"))  # pa [deg]
    for s, e, p in zip(sma, eps, pa):
        ax.add_patch(Ellipse((x0, y0), 2 * s, 2 * s * (1 - e), angle=p, fill=False, lw=0.7, color="tab:orange"))
    ax.set_title("fitted isophotes (one in every %d)" % every)


def show_profiles(a, eps, pa, b, a_min, a_ref=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.plot(a, eps, "o", ms=3, color="tab:blue")
    ax2.plot(a, pa, "o", ms=3, color="tab:blue")
    for ax in (ax1, ax2):
        ax.axvspan(0, a_min, color="0.9")
        ax.axvline(b["a_eps"], color="tab:purple", lw=1.5)
        if a_ref:
            ax.axvline(a_ref, color="k", ls="--", lw=0.8)
    ax1.axhline(0.25, color="0.5", ls=":", lw=0.8)
    ax1.set_ylabel("ε"), ax2.set_ylabel("PA [deg]"), ax2.set_xlabel("a [arcsec]")
    ax1.set_xlim(0, 3 * b["a_eps"])
    ax1.set_title("purple: bar semi-length (first peak of ε)   grey band: a < a_min")
    fig.tight_layout()


def show_modes(img, r, a0, am, bm, x0, y0, radii_px):
    """Brightness along circles and its Fourier reconstruction with m = 0 and m = 0…2."""
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    fig, axes = plt.subplots(1, len(radii_px), figsize=(5 * len(radii_px), 3.5), squeeze=False)
    from scipy.ndimage import map_coordinates
    for ax, rp in zip(axes[0], radii_px):
        i = int(np.argmin(np.abs(r - rp)))
        vals = map_coordinates(img, [y0 + r[i] * np.sin(phi), x0 + r[i] * np.cos(phi)], order=1)
        m2 = a0[i] + am[i, 1] * np.cos(2 * phi) + bm[i, 1] * np.sin(2 * phi)
        ax.plot(np.degrees(phi), vals, ".", ms=2, color="0.5", label="pixels")
        ax.axhline(a0[i], color="k", lw=1, label="m = 0 (round: bulge + disk)")
        ax.plot(np.degrees(phi), m2, color="tab:orange", lw=2, label="m = 0 + m = 2 (bar)")
        ax.set_xlabel("angle φ [deg]"), ax.set_title(f"circle r = {rp:.0f} px")
        ax.legend(fontsize=7)
    fig.tight_layout()


def show_bar_only(bo, x0, y0, half, angle_deg, a_bar_px, w_px):
    rot, cx, cy = bar_on_x(bo, x0, y0, angle_deg)
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = np.nanpercentile(np.abs(rot), 99.5)
    ax.imshow(rot, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.plot(*boxy(a_bar_px, w_px / 2, 0, cx, cy), color="tab:purple", lw=1.5)
    for f in (0.5, 0.75):
        for s in (-1, 1):
            ax.axvline(cx + s * f * a_bar_px, color="k", lw=0.6, ls=":")
    ax.set_xlim(cx - half, cx + half), ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("bar-only image (m = 2, 4, 6), bar on x\ndotted: where the width is measured")


def show_bar(img, res, pixscale, half, a_ref=None):
    """The measured bar on the image, and on the image rotated with the bar on x."""
    x0, y0 = res["x0"], res["y0"]
    angle = res["pa_bar"] + 90  # back to the image frame, from +x
    a, w = res["a_bar"] / pixscale, res["W_bar"] / pixscale
    rot, cx, cy = bar_on_x(img, x0, y0, angle)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    _show(ax[0], img, x0, y0, half)
    ax[0].plot(*boxy(a, w / 2, angle, x0, y0), color="tab:purple", lw=1.8)
    _show(ax[1], rot, cx, cy, half)
    ax[1].plot(*boxy(a, w / 2, 0, cx, cy), color="tab:purple", lw=1.8)
    if a_ref:
        for s in (-1, 1):
            ax[1].plot(cx + s * a_ref / pixscale, cy, "o", mfc="none", mec="k", ms=8)
    ax[0].set_title(f"a = {res['a_bar']:.1f}″, W = {res['W_bar']:.1f}″, PA = {res['pa_bar']:.0f}°")
    ax[1].set_title("bar on the x axis" + ("   circles: reference ends" if a_ref else ""))
    fig.tight_layout()


def show_bands(img, res, eps_prof, pixscale, a_ref, title=""):
    """The bar measured in r (blue) and in z (purple): on the r image rotated with the
    r bar on x, and on ε(a). res and eps_prof are {band: result}, {band: (a, ε)}."""
    colors = {"r": "tab:blue", "z": "tab:purple"}
    angle = res["r"]["pa_bar"] + 90
    rot, cx, cy = bar_on_x(img, res["r"]["x0"], res["r"]["y0"], angle)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.4]})
    _show(ax[0], rot, cx, cy, 1.6 * max(res["r"]["a_bar"], res["z"]["a_bar"], a_ref) / pixscale)
    for s in (-1, 1):
        ax[0].plot(cx + s * a_ref / pixscale, cy, "o", mfc="none", mec="k", ms=8)
    for b, r in res.items():
        turn = r["pa_bar"] + 90 - angle
        ax[0].plot(*boxy(r["a_bar"] / pixscale, r["W_bar"] / 2 / pixscale, turn, cx, cy), color=colors[b], lw=1.6)
        ax[1].plot(*eps_prof[b], ".", ms=3, color=colors[b], label=f"{b}: a = {r['a_bar']:.1f}″")
        ax[1].axvline(r["a_bar"], color=colors[b], lw=1)
    ax[1].axvline(a_ref, color="k", ls="--", lw=0.8, label=f"reference: {a_ref:.1f}″")
    ax[1].set(xlim=(0, 2.5 * max(r["a_bar"] for r in res.values())), xlabel="a [arcsec]", ylabel="ε")
    ax[1].legend(frameon=False)
    ax[0].set_title(title)
    fig.tight_layout()
