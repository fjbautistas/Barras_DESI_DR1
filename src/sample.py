"""
Parent sample of barred galaxies: Galaxy Zoo DESI (Walmsley et al. 2023)
votes matched by position to the DESI DR1 spectroscopic catalogue.
"""
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

GZ_COLUMNS = ["dr8_id", "ra", "dec", "bar_strong_fraction", "bar_weak_fraction"]
AUX_COLUMNS = ["TARGETID", "SHAPE_R", "SHAPE_E1", "SHAPE_E2", "MORPHTYPE"]


def load_gz_bars(path, pbar_min):
    """GZ DESI galaxies with p_bar = p_strong + p_weak > pbar_min.

    The bar fractions only exist for galaxies that reached the bar question
    (featured and not edge-on), so no other morphological cut is needed.
    """
    gz = pd.read_parquet(path, columns=GZ_COLUMNS)
    gz["pbar"] = gz["bar_strong_fraction"] + gz["bar_weak_fraction"]
    return gz[gz["pbar"] > pbar_min].reset_index(drop=True)


def match_desi(gz, base, tol_arcsec):
    """Nearest DESI galaxy to each GZ entry, kept if closer than tol_arcsec.

    Two GZ entries can fall on the same DESI galaxy; the closer one is kept.
    """
    c_gz = SkyCoord(gz["ra"].to_numpy() * u.deg, gz["dec"].to_numpy() * u.deg)
    c_desi = SkyCoord(base["RA"].to_numpy() * u.deg, base["DEC"].to_numpy() * u.deg)
    idx, sep, _ = c_gz.match_to_catalog_sky(c_desi)
    sep = sep.arcsec
    ok = sep < tol_arcsec

    out = base.iloc[idx[ok]].reset_index(drop=True)
    gz_ok = gz[ok].reset_index(drop=True)
    out = pd.concat([out, gz_ok.drop(columns=["ra", "dec"])], axis=1)
    out["sep_arcsec"] = sep[ok]

    out = out.sort_values("sep_arcsec").drop_duplicates("TARGETID")
    return out.sort_values("TARGETID").reset_index(drop=True)


def add_shape(sample, aux_path):
    """Legacy Survey shape (half-light radius, ellipticity components)."""
    aux = pd.read_parquet(aux_path, columns=AUX_COLUMNS)
    out = sample.merge(aux, on="TARGETID", how="left")
    if out["SHAPE_R"].isna().any():
        raise ValueError(f"{out['SHAPE_R'].isna().sum()} galaxies without Legacy shape")
    return out
