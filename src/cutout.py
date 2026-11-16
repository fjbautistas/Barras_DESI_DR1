"""
Cutouts from the Legacy Survey viewer: FITS (flux, nanomaggies) to measure,
JPG to look at. The viewer also serves SDSS images, resampled at their
native scale, so both surveys go through the same code. The size of the
cutout comes from the Tractor SHAPE_R, queried in Astro Data Lab when the
sample does not have it.
"""
import io
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

VIEWER = "https://www.legacysurvey.org/viewer"
SURVEYS = {  # layer, native pixel scale [arcsec/px]
    "desi": ("ls-dr10", 0.262),
    "sdss": ("sdss", 0.396),
}
ENDPOINT = {"fits": "fits-cutout", "jpg": "jpeg-cutout"}
MAGIC = {"fits": b"SIMPLE", "jpg": b"\xff\xd8"}  # the viewer answers errors with HTML
TAP = "https://datalab.noirlab.edu/tap/sync"  # accepts boxes joined by OR, not CIRCLE nor uploads


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


def tractor_shape(ra, dec, radius=1.0, batch=100):
    """SHAPE_R [arcsec] and TYPE of the nearest Legacy DR10 Tractor source within radius [arcsec]
    of each position; NaN and None where there is none. DUP rows (shapeless copies of an atlas
    galaxy) are skipped."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    out = pd.DataFrame({"SHAPE_R": np.nan, "TYPE": None}, index=range(len(ra)))
    d = radius / 3600
    for s in range(0, len(ra), batch):
        r, de = ra[s:s + batch], dec[s:s + batch]
        dr = d / np.cos(np.radians(de))
        boxes = " OR ".join(f"(ra BETWEEN {a - x:.7f} AND {a + x:.7f} AND dec BETWEEN {b - d:.7f} AND {b + d:.7f})"
                            for a, b, x in zip(r, de, dr))
        query = f"SELECT ra, dec, shape_r, type FROM ls_dr10.tractor WHERE type <> 'DUP' AND ({boxes})"
        body = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}).encode()
        cat = pd.read_csv(io.StringIO(urlopen(TAP, body, timeout=300).read().decode()))
        if cat.empty:
            continue
        i, sep, _ = SkyCoord(r * u.deg, de * u.deg).match_to_catalog_sky(SkyCoord(cat.ra.values * u.deg, cat.dec.values * u.deg))
        ok = sep.arcsec <= radius
        out.loc[s + np.flatnonzero(ok), ["SHAPE_R", "TYPE"]] = cat.iloc[i[ok]][["shape_r", "type"]].values
    return out
