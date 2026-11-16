# checks/

Validation done while building the method (11–12 Nov 2026). Not needed to
measure bars; kept as evidence and to repeat the tests.

- `bar_full.py`, `fourier_full.py`, `measure_full.py`: the full versions of
  `src/bar.py`, `src/fourier.py`, `src/measure.py` before the simplification
  (deprojection, Fourier bar length by the φ2 phase and the intensity ratio,
  a_up). The scripts below import them.
- `decompose.py`, `decompose_bars.py`, `plot_decomp.py`: 2D decomposition
  bulge + disk + bar + ring with pyimfit (angle check on large bars).
- `plot_profiles.py`, `plot_fourier.py`: ε(a), PA(a) and Fourier diagnostics
  of one galaxy.
- `jpg_vs_fits.py`: the same method on the FITS and on the colour JPG of one
  galaxy (the JPG bar comes out short).
- `no_calzan.py`: the 24 galaxies that failed when the SDSS-trained method of
  the reference was applied to DESI, against the rest of the 448.
- `build_common.py`, `plot_vs_reference.py`: first comparison with the 157
  galaxies in common with the reference (superseded by
  `scripts/build_reference_sample.py` and `tools/compare_reference.py`).

Run from the repository root, e.g. `python -m checks.decompose_bars ...`.
