from pathlib import Path
import numpy as np
from .base import Instrument

# NEON AOP spectral metadata source
# ---------------------------------------------------------------------------
# NEON AOP reflectance HDF5 files (data product DP3.30006.001) embed per-tile
# wavelength centers and FWHM in the group:
#   /{site_code}/Reflectance/Metadata/Spectral_Data/
#     Wavelength  — band center wavelengths (nm)
#     FWHM        — full width at half maximum per band (nm)
#
# The site code (e.g. 'HARV', 'SERC') is the first top-level key in the file
# and varies by collection site.  Reflectance data lives at:
#   /{site_code}/Reflectance/Reflectance_Data  (rows × cols × bands, int16)
# with a Scale_Factor attribute (typically 10000) to convert to 0–1 reflectance.
#
# NEON data portal: https://data.neonscience.org/  (product DP3.30006.001)
# HDF5 structure confirmed via EnSpec/hytools io/neon.py:
#   https://github.com/EnSpec/hytools/blob/master/hytools/io/neon.py
# ---------------------------------------------------------------------------


class NEON(Instrument):
    """
    NEON AOP (Airborne Observation Platform) hyperspectral instrument.

    Parameters
    ----------
    wavelengths : (n_bands,) array of band center wavelengths in nm
    fwhm : (n_bands,) array of FWHM values in nm. Defaults to 5.0 nm if not provided.
    h5_file : path to a NEON AOP reflectance HDF5 file. If given, wavelengths and
              FWHM are read from the file and the wavelengths/fwhm params are ignored.
    """

    def __init__(
        self,
        wavelengths: np.ndarray | None = None,
        fwhm: np.ndarray | None = None,
        h5_file: str | None = None,
    ) -> None:
        if h5_file is not None:
            self._wavelengths, self._fwhm = _read_spectral_from_h5(h5_file)
        elif wavelengths is not None:
            self._wavelengths = np.asarray(wavelengths, dtype=float)
            self._fwhm = (
                np.asarray(fwhm, dtype=float)
                if fwhm is not None
                else np.full(len(self._wavelengths), 5.0)
            )
        else:
            raise ValueError("Provide either 'h5_file' or 'wavelengths'.")

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    @property
    def fwhm(self) -> np.ndarray:
        return self._fwhm


def _read_spectral_from_h5(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py  # type: ignore
    except ImportError as e:
        raise ImportError("h5py is required to read NEON HDF5 files: pip install h5py") from e

    with h5py.File(filepath, "r") as f:
        site = next(iter(f.keys()))
        spec = f[site]["Reflectance"]["Metadata"]["Spectral_Data"]
        wl = spec["Wavelength"][:]
        fwhm = spec["FWHM"][:]
    return wl.astype(float), fwhm.astype(float)
