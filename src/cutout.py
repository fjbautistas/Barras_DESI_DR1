"""
Cutouts from the Legacy Survey viewer: FITS (flux, nanomaggies) to measure,
JPG to look at. The viewer also serves SDSS images, resampled at their
native scale, so both surveys go through the same code.
"""
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlretrieve

import numpy as np

VIEWER = "https://www.legacysurvey.org/viewer"
SURVEYS = {  # layer, native pixel scale [arcsec/px]
    "desi": ("ls-dr10", 0.262),
    "sdss": ("sdss", 0.396),
}
ENDPOINT = {"fits": "fits-cutout", "jpg": "jpeg-cutout"}
MAGIC = {"fits": b"SIMPLE", "jpg": b"\xff\xd8"}  # the viewer answers errors with HTML


def pixscale(survey):
    return SURVEYS[survey][1]


def cutout_size(shape_r, survey="desi", n_re=4.0, min_px=128, max_px=1024):
    """Side in px: the field reaches n_re half-light radii from the centre."""
    side = 2 * n_re * np.asarray(shape_r) / pixscale(survey)
    return np.clip(np.round(side), min_px, max_px).astype(int)


def cutout_url(ra, dec, size, survey="desi", kind="fits", band="r"):
    layer, scale = SURVEYS[survey]
    url = (f"{VIEWER}/{ENDPOINT[kind]}?ra={ra:.6f}&dec={dec:.6f}"
           f"&layer={layer}&pixscale={scale}&size={size}")
    return url + (f"&bands={band}" if kind == "fits" else "")


def cutout_paths(out_dir, name, survey="desi", band="r"):
    out_dir = Path(out_dir)
    return {"fits": out_dir / f"{name}_{survey}_{band}.fits", "jpg": out_dir / f"{name}_{survey}.jpg"}


def _fetch(url, path, tries=6, wait=5.0):
    """Download with retries: the viewer answers 429 when queried too fast."""
    for k in range(tries):
        try:
            urlretrieve(url, path)
            return
        except HTTPError as err:
            if err.code != 429 or k == tries - 1:
                raise
            time.sleep(wait * 2**k)


def download(ra, dec, size, out_dir, name, survey="desi", band="r"):
    """FITS in one band and colour JPG of the same field; skips existing files."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = cutout_paths(out_dir, name, survey, band)
    for kind, path in paths.items():
        if not path.exists():
            _fetch(cutout_url(ra, dec, size, survey, kind, band), path)
        if not path.read_bytes()[:6].startswith(MAGIC[kind]):
            path.unlink()
            raise RuntimeError(f"{name}: the viewer did not return a {kind}")
    return paths
