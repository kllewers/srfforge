from __future__ import annotations

import numpy as np
from .instruments.base import Instrument
from .srf import build_convolution_matrix


class BandConvolver:
    """
    Convolve reflectance data from one instrument's spectral sampling to another's.

    Uses Gaussian spectral response functions parameterized by the target instrument's
    band center wavelengths and FWHM.

    Parameters
    ----------
    source : Instrument
        Source instrument (defines spectral sampling of the input data).
    target : Instrument
        Target instrument (defines output spectral sampling).

    Examples
    --------
    Array-based usage::

        from srfforge import BandConvolver
        from srfforge.instruments import NEON, EMIT

        neon = NEON(wavelengths=my_wl, fwhm=my_fwhm)
        emit = EMIT()
        convolve = BandConvolver(source=neon, target=emit)
        emit_refl = convolve(neon_refl)  # (..., 288)

    File-based usage::

        from srfforge.io import read_neon_h5

        refl, neon, meta = read_neon_h5("path/to/neon.h5")
        convolve = BandConvolver(source=neon, target=EMIT())
        emit_refl = convolve(refl)
    """

    def __init__(self, source: Instrument, target: Instrument) -> None:
        self.source = source
        self.target = target
        self._matrix = build_convolution_matrix(
            source.wavelengths, target.wavelengths, target.fwhm
        )

    @property
    def matrix(self) -> np.ndarray:
        """Convolution matrix of shape (n_target_bands, n_source_bands)."""
        return self._matrix

    def __call__(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Convolve reflectance to target instrument bands.

        Parameters
        ----------
        reflectance : array of shape (..., n_source_bands)

        Returns
        -------
        array of shape (..., n_target_bands)
        """
        return reflectance @ self._matrix.T

    def __repr__(self) -> str:
        return (
            f"BandConvolver(\n"
            f"  source={self.source!r},\n"
            f"  target={self.target!r}\n"
            f")"
        )
