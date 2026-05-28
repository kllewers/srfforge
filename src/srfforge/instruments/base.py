from abc import ABC, abstractmethod
import numpy as np


class Instrument(ABC):
    @property
    @abstractmethod
    def wavelengths(self) -> np.ndarray:
        """Band center wavelengths in nm, ascending order."""
        ...

    @property
    @abstractmethod
    def fwhm(self) -> np.ndarray:
        """FWHM for each band in nm."""
        ...

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_bands={len(self.wavelengths)}, "
            f"range={self.wavelengths[0]:.1f}–{self.wavelengths[-1]:.1f} nm)"
        )
