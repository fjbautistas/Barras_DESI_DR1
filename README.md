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

1. **Image**: a FITS cutout in one band (Legacy Survey DR10 r or z, 0.262″/px),
   centred on the galaxy, reaching 4 half-light radii (Tractor `SHAPE_R`).
2. **Sky and masks**: sky level by sigma clipping; every source except the
   central galaxy is masked.
3. **Isophotes**: an ellipse is fitted at each radius on its own, starting
   from the last converged one (photutils); ε(a) and PA(a).
4. **Length and orientation**: the bar is the first peak of ε ≥ 0.25 followed
   by a drop of 0.1 (Marinova & Jogee 2007), searched beyond
   a_min = max(2″, 1 kpc). Its semi-major axis is the bar semi-length `a_bar`
   (Erwin 2005) and its PA the bar orientation.
5. **Width**: the light is split in angular Fourier modes; the even modes
   m = 2, 4, 6 hold the bar and none of the round bulge or disk. The width
   `W_bar` is the FWHM of that bulge-free image across the bar, at 0.5 and
   0.75 of its length.
6. **Disc and deprojection**: disc ε and PA from the isophotes beyond
   1.5 `a_bar` with 23.5 < μ_r < 25; sizes are given projected and on the
   disc plane.
7. **Physical size**: `L_kpc = 2 a_bar`, `W_kpc`, with z and Planck18.

Validation against the 448 bars of Alonso et al. (2024): `a_bar` / reference
= 1.01 ± 0.10, 88 % within 20 %, the same for the 157 galaxies used while
building the method and the 291 never looked at
(`outputs/gallery/reference_1to1.png`). The reference widths include the
bulge; ours do not, so `W_bar` is about half of them. `b_iso` (the
isophote semi-minor axis) is the comparable one. The deprojection agrees
with S⁴G orientations (factor ratio 0.98, 93 % within 20 %). The z band gives
the same agreement with the reference (88.6 %) and the same deprojection
(0.98), with the same disc window; r is deeper, z follows the stellar mass.
Measuring both and flagging |r/z − 1| > 0.15 picks out galaxies to check. Details, tests
and the error of the length: `docs/barras.pdf`.

## Use

```bash
pip install -r requirements.txt     # reference environment: conda astro-clean
```

One image:

```python
from src.measure import measure_fits, measure_jpg
bar = measure_fits("galaxy_r.fits", z=0.039)       # pixel scale from the WCS
bar = measure_jpg("galaxy.jpg", 0.262, z=0.039)    # last resort: the bar comes out ~25 % short
bar["a_bar"], bar["W_bar"], bar["pa_bar"], bar["L_kpc"]
```

A list of galaxies: only RA and DEC are needed. `build_input.py` adds the
missing TARGETID (row number), Z (NaN: sizes in arcsec only) and `SHAPE_R`
(nearest Legacy DR10 Tractor source within 1″, from Astro Data Lab); then the
cutouts are downloaded and every galaxy measured.

```bash
python scripts/build_input.py --list galaxies.csv --output outputs/sample/input.parquet
python scripts/measure_bars.py --sample outputs/sample/input.parquet \
    --output outputs/bars/bars.parquet --n-jobs 8            # --band z to measure in z
python tools/plot_gallery.py --bars outputs/bars/bars.parquet --out-dir outputs/gallery/all --rotate
```

On a cluster: build the input and download on a machine with internet,
then one job per block, then join the blocks.

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
| `survey`, `band` | images used (desi or sdss; r or z) |
| `a_bar` | bar semi-length [″] |
| `W_bar` | bar full width, bulge-free [″] |
| `pa_bar` | bar PA [deg, north through east] |
| `eps_bar` | ellipticity of the isophote at `a_bar` |
| `b_iso` | semi-minor axis of that isophote [″] (includes bulge light) |
| `L_kpc`, `W_kpc` | 2 `a_bar` and `W_bar` in kpc (projected; NaN without z) |
| `a_bar_dep`, `W_bar_dep`, `L_kpc_dep`, `W_kpc_dep` | the same on the disc plane |
| `eps_disc`, `pa_disc`, `dphi` | disc ellipticity and PA, angle between bar and disc [deg] |
| `a_min` | radius beyond which the bar was searched [″] |
| `flag_eps` | ε_bar ≥ 0.25 |
| `flag_end` | ε drops or PA turns beyond `a_bar`: the bar ends |
| `flag_pa` | PA constant over the outer half of the bar |
| `x0`, `y0`, `sky`, `sky_std`, `a_max_iso` | centre [px], sky, its noise, largest isophote [″] |

## Files

| File | Content |
|---|---|
| `src/cutout.py` | Legacy viewer cutouts (FITS, JPG; DESI or SDSS) and `SHAPE_R` from Tractor |
| `src/isophotes.py` | sky, masks, galaxy centre and the radius-by-radius isophote fit |
| `src/bar.py` | the bar in ε(a), PA(a): first peak, flags; kpc per arcsec |
| `src/fourier.py` | Fourier modes on circles, the bulge-free bar image and its width |
| `src/measure.py` | one galaxy end to end: disc, deprojection, output dictionary |
| `src/sample.py` | GZ DESI bars matched to DESI DR1, with the Legacy shape |
| `src/utils.py` | provenance `.json` of every product |
| `scripts/build_sample.py` | parent sample (27 283 barred galaxies) |
| `scripts/build_reference_sample.py` | the 448 galaxies of the reference, as a sample |
| `scripts/build_input.py` | any list with RA, DEC into a sample for `measure_bars.py` |
| `scripts/measure_bars.py` | measure a sample, in parallel or in cluster blocks |
| `scripts/merge_bars.py` | join the cluster blocks |
| `tools/plot_gallery.py` | gallery of measured bars over the images |
| `tools/compare_reference.py` | our bars against the reference |
| `tools/compare_surveys.py` | method vs data: DESI, SDSS and the reference |
| `tools/steps.py` | figures of each step (used by the notebook) and drawing helpers |
| `notebooks/Barra_ejemplo.ipynb` | the method step by step on one galaxy |
| `checks/` | validation done while building the method (see `checks/README.md`) |
| `docs/barras.tex`, `docs/barras.pdf` | the method explained, with every test (in Spanish) |
| `docs/scripts_barras/` | one script per figure of the document |
| `modelo/` | reference measurements and code of Alonso et al. (2024) (not tracked) |

## Samples

| Script | Product |
|---|---|
| `scripts/build_sample.py` | `outputs/sample/barred_sample.parquet`: Galaxy Zoo DESI p_bar > 0.5, DESI DR1 within 1″, z < 0.1 (27 283) |
| `scripts/build_reference_sample.py` | `outputs/sample/reference_all.parquet`: the 448 galaxies of the reference |

## Open

- Resolution limit: most bars of the full sample are 2–5″, close to the
  seeing; 2″ is still provisional (1 kpc only fixes nearby galaxies).
- The error of the length (range of radii where ε is consistent with the
  peak, `docs/barras.pdf`) is tested but not in `src/` yet.
- Stars on top of the galaxy and artefacts are not masked (GLIDErS?).
