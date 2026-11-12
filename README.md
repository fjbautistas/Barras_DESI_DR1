# Bars in DESI DR1

Length, width and orientation of galactic bars from a single-band image,
with an analytic method (no training, no hand marks).

```text
Barras_DESI_DR1/
├── src/        library: cutouts, isophotes, bar, Fourier width, measure
├── scripts/    samples and the measurement of a whole sample
├── tools/      gallery and comparison with a reference
├── checks/     validation done while building the method (not needed to measure)
├── modelo/     reference measurements and code of Alonso et al. (2024)
└── outputs/    samples, bar tables, cutouts, figures
```

## Method

1. **Image**: a FITS cutout in one band (Legacy Survey DR10 r, 0.262″/px),
   centred on the galaxy, reaching 4 half-light radii.
2. **Sky and masks**: sky level by sigma clipping; every source except the
   central galaxy is masked.
3. **Isophotes**: an ellipse is fitted at each radius on its own, starting
   from the last converged one (photutils); ε(a) and PA(a).
4. **Length and orientation**: the bar is the first peak of ε ≥ 0.25 followed
   by a drop of 0.1 (Marinova & Jogee 2007). Its semi-major axis is the bar
   semi-length `a_bar` (Erwin 2005) and its PA the bar orientation.
5. **Width**: the light is split in angular Fourier modes; the even modes
   m = 2, 4, 6 hold the bar and none of the round bulge or disk. The width
   `W_bar` is the FWHM of that bulge-free image across the bar, at 0.5 and
   0.75 of its length.
6. **Physical size**: `L_kpc = 2 a_bar`, `W_kpc`, with z and Planck18.
   All sizes are projected on the sky.

Validation against the 448 bars of Alonso et al. (2024): `a_bar` / reference
= 1.01 ± 0.10, 88 % within 20 %, the same for the 157 galaxies used while
building the method and the 291 never looked at
(`outputs/gallery/reference_1to1.png`). The reference widths include the
bulge; ours do not, so `W_bar` is about half of them. `b_iso` (the
isophote semi-minor axis) is the comparable one.

## Use

```bash
pip install -r requirements.txt     # reference environment: conda astro-clean
```

One image:

```python
from src.measure import measure_fits
bar = measure_fits("galaxy_r.fits", z=0.039)   # pixel scale from the WCS
bar["a_bar"], bar["W_bar"], bar["pa_bar"], bar["L_kpc"]
```

A sample (parquet with TARGETID, RA, DEC, Z, SHAPE_R): the cutouts are
downloaded and every galaxy measured.

```bash
python scripts/measure_bars.py --sample outputs/sample/barred_sample.parquet \
    --output outputs/bars/bars.parquet --n-jobs 8
python tools/plot_gallery.py --bars outputs/bars/bars.parquet --out-dir outputs/gallery/all --rotate
```

On a cluster: download first on a node with internet, then one job per
block, then join the blocks.

```bash
python scripts/measure_bars.py --sample S --output outputs/bars/bars.parquet --download-only --n-jobs 4
python scripts/measure_bars.py --sample S --output outputs/bars/bars.parquet \
    --n-chunks 100 --chunk $SLURM_ARRAY_TASK_ID --n-jobs $SLURM_CPUS_PER_TASK
python scripts/merge_bars.py --output outputs/bars/bars.parquet
```

Each product gets a `.json` with inputs, parameters and counts (including
the galaxies that failed and why).

## Output columns

| Column | Meaning |
|---|---|
| `a_bar` | bar semi-length [″] |
| `W_bar` | bar full width, bulge-free [″] |
| `pa_bar` | bar PA [deg, north through east] |
| `eps_bar` | ellipticity of the isophote at `a_bar` |
| `b_iso` | semi-minor axis of that isophote [″] (includes bulge light) |
| `L_kpc`, `W_kpc` | 2 `a_bar` and `W_bar` in kpc (projected) |
| `flag_eps` | ε_bar ≥ 0.25 |
| `flag_end` | ε drops or PA turns beyond `a_bar`: the bar ends |
| `flag_pa` | PA constant over the outer half of the bar |
| `x0`, `y0`, `sky`, `sky_std`, `a_max_iso` | centre [px], sky, its noise, largest isophote [″] |

## Samples

| Script | Product |
|---|---|
| `scripts/build_sample.py` | `outputs/sample/barred_sample.parquet`: Galaxy Zoo DESI p_bar > 0.5, DESI DR1 within 1″, z < 0.1 (27 283) |
| `scripts/build_reference_sample.py` | `outputs/sample/reference_all.parquet`: the 448 galaxies of the reference |

## Open

- Resolution limit: most bars of the full sample are 2–5″, close to the
  seeing; the smallest measurable bar (`A_MIN`, now 2″) has to be fixed.
- Stars on top of the galaxy and artefacts are not masked (GLIDErS?).
- Deprojection needs a reliable disk orientation; sizes are projected.
