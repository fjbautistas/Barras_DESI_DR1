# checks/

Validation done while building the method (11–14 Nov 2026). Not needed to
measure bars; kept because the figures and tests of `docs/barras.pdf` use it.

- `bar_full.py`, `fourier_full.py`: the full versions of `src/bar.py` and
  `src/fourier.py` before the simplification (upper bound a_up, Fourier bar
  length by the φ2 phase). The figures in `docs/scripts_barras/` import them.
- `decompose.py`, `decompose_bars.py`: 2D decomposition bulge + disk + bar +
  ring with pyimfit (angle check on large bars, section 11 of the document).

Run from the repository root, e.g. `python -m checks.decompose_bars ...`.
