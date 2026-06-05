"""
Visualization helpers for srfforge band-convolution results.

Install plot dependencies:
    pip install 'srfforge[plot]'

Example
-------
    from srfforge.plot import compare_spectra, rgb_quicklook

    fig = compare_spectra(neon_refl, emit_refl, source=neon, target=emit)
    fig = rgb_quicklook(neon_refl, instrument=neon)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Module: plot/__init__.py  —  public API for visualization
#
#   called by ◄── user code
#   calls     ──► plot/core.py
# ─────────────────────────────────────────────────────────────────────────────

from .core import compare_spectra, rgb_quicklook, plot_residuals

__all__ = ["compare_spectra", "rgb_quicklook", "plot_residuals"]
