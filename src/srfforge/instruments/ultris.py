import numpy as np
from .base import Instrument

# Cubert Ultris VNIR spectral metadata
# ---------------------------------------------------------------------------
# 164 bands, 350–1002 nm, uniform 4 nm sampling interval.
# Source: Cubert_Ultris_VNIR_spectral_bands.xlsx (provided by manufacturer).
#
# FWHM is not published in the band table; the constructor accepts a
# constant FWHM value (default 8 nm — verify against your instrument's
# calibration report before use in precision applications).
# ---------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# Module: instruments/ultris.py  —  Cubert Ultris VNIR instrument
#
#   called by ◄── user code / BandConvolver (via .wavelengths / .fwhm)
#   calls     ──► (none — bundled wavelengths, no file needed)
# ─────────────────────────────────────────────────────────────────────────────

_ULTRIS_WAVELENGTHS = np.arange(350, 1003, 4, dtype=float)  # 164 bands, 350–1002 nm


#--------------------------------------
# Called by: user code
# Calls:     none
#--------------------------------------
class CubertUltris(Instrument):
    """
    Cubert Ultris VNIR snapshot hyperspectral imager.

    164 bands, 350–1002 nm, 4 nm uniform sampling.
    Wavelengths are bundled from the manufacturer's band table.

    Parameters
    ----------
    fwhm_nm : per-band FWHM in nm. Defaults to 8 nm — override with your
              instrument's calibration value if precision matters.
    """

    def __init__(self, fwhm_nm: float = 8.0) -> None:
        self._wavelengths = _ULTRIS_WAVELENGTHS.copy()
        self._fwhm = np.full(len(self._wavelengths), float(fwhm_nm))

    #--------------------------------------
    # Called by: BandConvolver.__init__ (convolve.py)
    # Calls:     none
    #--------------------------------------
    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    #--------------------------------------
    # Called by: BandConvolver.__init__ (convolve.py)
    # Calls:     none
    #--------------------------------------
    @property
    def fwhm(self) -> np.ndarray:
        return self._fwhm
