from abc import ABC, abstractmethod
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Module: instruments/base.py  —  abstract Instrument base class
#
#   called by ◄── convolve.py::BandConvolver (accesses .wavelengths / .fwhm)
#   calls     ──► (none — interface only)
#   implemented by: NEON (neon.py), EMIT (emit.py), AVIRIS3 (aviris3.py)
# ─────────────────────────────────────────────────────────────────────────────


#--------------------------------------
# Called by: BandConvolver.__init__ (convolve.py), via .wavelengths / .fwhm
# Calls:     none
#--------------------------------------
class Instrument(ABC):
    #--------------------------------------
    # Called by: BandConvolver.__init__ (convolve.py)
    # Calls:     none
    #--------------------------------------
    @property
    @abstractmethod
    def wavelengths(self) -> np.ndarray:
        """Band center wavelengths in nm, ascending order."""
        ...

    #--------------------------------------
    # Called by: BandConvolver.__init__ (convolve.py)
    # Calls:     none
    #--------------------------------------
    @property
    @abstractmethod
    def fwhm(self) -> np.ndarray:
        """FWHM for each band in nm."""
        ...

    #--------------------------------------
    # Called by: Python repr() / print()
    # Calls:     none
    #--------------------------------------
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_bands={len(self.wavelengths)}, "
            f"range={self.wavelengths[0]:.1f}–{self.wavelengths[-1]:.1f} nm)"
        )
