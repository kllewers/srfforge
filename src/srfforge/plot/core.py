"""Spectrum comparison, RGB quicklook, and residual plots."""

# ─────────────────────────────────────────────────────────────────────────────
# Module: plot/core.py  —  matplotlib-based visualization
#
#   called by ◄── plot/__init__.py (re-exports)
#   calls     ──► matplotlib  (optional dependency)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from ..instruments.base import Instrument


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting: pip install 'srfforge[plot]'"
        ) from e


#--------------------------------------
# Called by: user code
# Calls:     matplotlib.pyplot
#--------------------------------------
def compare_spectra(
    source_refl: np.ndarray,
    target_refl: np.ndarray,
    source: "Instrument",
    target: "Instrument",
    pixel: tuple[int, int] | None = None,
    n_sample: int = 200,
    source_label: str | None = None,
    target_label: str | None = None,
    ax=None,
) -> "plt.Figure":
    """
    Plot mean spectra of source and convolved target reflectance arrays side-by-side.

    Parameters
    ----------
    source_refl : (..., n_source_bands), float, values 0–1
    target_refl : (..., n_target_bands), float, values 0–1
    source : source Instrument (provides wavelengths for x-axis)
    target : target Instrument (provides wavelengths for x-axis)
    pixel : optional (row, col) — plot a single pixel instead of a spatial mean
    n_sample : number of random pixels to average when pixel is None
    source_label : legend label for source (defaults to instrument class name)
    target_label : legend label for target (defaults to instrument class name)
    ax : optional existing matplotlib Axes

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_matplotlib()

    src_wl = np.asarray(source.wavelengths)
    tgt_wl = np.asarray(target.wavelengths)

    src_flat = source_refl.reshape(-1, source_refl.shape[-1])
    tgt_flat = target_refl.reshape(-1, target_refl.shape[-1])

    if pixel is not None:
        row, col = pixel
        src_spec = source_refl[row, col]
        tgt_spec = target_refl[row, col]
        title_suffix = f"pixel ({row}, {col})"
    else:
        # Only sample from pixels that are fully valid (no NaN in any band)
        valid = np.isfinite(src_flat).all(axis=1)
        n_valid = valid.sum()
        if n_valid == 0:
            raise ValueError("No valid (non-NaN) pixels found in source_refl.")
        valid_idx = np.where(valid)[0]
        rng = np.random.default_rng(seed=0)
        n = min(n_sample, n_valid)
        idx = rng.choice(valid_idx, size=n, replace=False)
        src_spec = src_flat[idx].mean(axis=0)
        tgt_spec = tgt_flat[idx].mean(axis=0)
        title_suffix = f"mean of {n} valid pixels (of {n_valid:,} total)"

    fig, ax_out = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 4))

    src_name = source_label or type(source).__name__
    tgt_name = target_label or type(target).__name__

    ax_out.plot(src_wl, src_spec, lw=1.2, alpha=0.9, label=src_name)
    ax_out.plot(tgt_wl, tgt_spec, lw=1.5, ls="--", alpha=0.9, label=f"{tgt_name} (convolved)")
    ax_out.set_xlabel("Wavelength (nm)")
    ax_out.set_ylabel("Reflectance")
    ax_out.set_title(f"Spectral comparison — {title_suffix}")
    ax_out.legend()
    ax_out.grid(True, alpha=0.3)
    fig = ax_out.figure
    fig.tight_layout()
    return fig


#--------------------------------------
# Called by: user code
# Calls:     matplotlib.pyplot
#--------------------------------------
def rgb_quicklook(
    refl: np.ndarray,
    instrument: "Instrument",
    rgb_nm: tuple[float, float, float] = (640.0, 550.0, 460.0),
    percentile: tuple[float, float] = (2.0, 98.0),
    aspect: str = "equal",
    ax=None,
) -> "plt.Figure":
    """
    Display a 3-band RGB composite from reflectance data.

    Parameters
    ----------
    refl : (rows, cols, n_bands), float, values 0–1
    instrument : Instrument providing band center wavelengths
    rgb_nm : (R, G, B) wavelength targets in nm
    percentile : (low, high) for contrast stretch
    ax : optional existing matplotlib Axes

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_matplotlib()

    wl = np.asarray(instrument.wavelengths)
    band_indices = tuple(int(np.argmin(np.abs(wl - nm))) for nm in rgb_nm)

    rgb = refl[..., band_indices].astype(np.float32)

    lo = np.nanpercentile(rgb, percentile[0])
    hi = np.nanpercentile(rgb, percentile[1])
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    # NaN pixels → white so they don't look like valid dark features
    rgb = np.where(np.isnan(rgb), 1.0, rgb)

    fig, ax_out = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 6))
    ax_out.imshow(rgb, aspect=aspect)
    ax_out.set_title(
        f"{type(instrument).__name__} RGB  "
        f"({rgb_nm[0]:.0f} / {rgb_nm[1]:.0f} / {rgb_nm[2]:.0f} nm)"
    )
    ax_out.axis("off")
    fig = ax_out.figure
    fig.tight_layout()
    return fig


#--------------------------------------
# Called by: user code
# Calls:     matplotlib.pyplot
#--------------------------------------
def plot_residuals(
    source_refl: np.ndarray,
    target_refl: np.ndarray,
    source: "Instrument",
    target: "Instrument",
    n_sample: int = 200,
    ax=None,
) -> "plt.Figure":
    """
    Compare SRF convolution vs linear interpolation on the same source spectra.

    Shows where proper Gaussian SRF convolution differs from naive linear
    interpolation — typically at steep spectral gradients near water vapour
    absorption features (~1340, ~1800 nm) and the O₂-A band (~760 nm).

    Parameters
    ----------
    source_refl : (..., n_source_bands)
    target_refl : (..., n_target_bands) — already SRF-convolved from source
    source : source Instrument
    target : target Instrument
    n_sample : number of random pixels to use
    ax : optional existing matplotlib Axes

    Returns
    -------
    matplotlib Figure
    """
    plt = _require_matplotlib()

    src_flat = source_refl.reshape(-1, source_refl.shape[-1])
    tgt_flat = target_refl.reshape(-1, target_refl.shape[-1])

    valid = np.isfinite(src_flat).all(axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise ValueError("No valid (non-NaN) pixels found in source_refl.")
    rng = np.random.default_rng(seed=42)
    n = min(n_sample, len(valid_idx))
    idx = rng.choice(valid_idx, size=n, replace=False)

    src_mean = src_flat[idx].mean(axis=0)
    tgt_mean = tgt_flat[idx].mean(axis=0)

    # Reference: naive linear interpolation from source to target wavelengths
    src_wl = np.asarray(source.wavelengths)
    tgt_wl = np.asarray(target.wavelengths)
    tgt_interp = np.interp(tgt_wl, src_wl, src_mean)

    residual = tgt_mean - tgt_interp

    fig, ax_out = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 3))
    ax_out.plot(tgt_wl, residual, lw=1.2, color="purple")
    ax_out.axhline(0, color="k", lw=0.5)
    ax_out.fill_between(tgt_wl, 0, residual, alpha=0.2, color="purple")
    ax_out.set_xlabel("Wavelength (nm)")
    ax_out.set_ylabel("SRF conv − linear interp")
    ax_out.set_title(
        f"{type(source).__name__} → {type(target).__name__}  |  "
        f"max |residual| = {np.nanmax(np.abs(residual)):.4f}"
    )
    ax_out.grid(True, alpha=0.3)
    fig = ax_out.figure
    fig.tight_layout()
    return fig
