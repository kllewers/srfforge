from pathlib import Path
import numpy as np
from .base import Instrument

# AVIRIS-3 spectral metadata source
# ---------------------------------------------------------------------------
# Specs: 285 bands, 390–2500 nm, ~7.4 nm spectral sampling, ~7.4 nm FWHM.
# Distributed as NetCDF4 (L1B/L2A) via ORNL DAAC (doi:10.3334/ORNLDAAC/2356).
#
# There is NO standalone wavelength/FWHM file (unlike EMIT's .txt).
# The per-band wavelength and FWHM are embedded inside each product file
# under the top-level group that matches the data type:
#   nc4_obj['radiance']['wavelength']   — center wavelengths (nm)
#   nc4_obj['radiance']['fwhm']         — FWHM per band (nm)
# For L2A reflectance files, replace 'radiance' with 'reflectance'.
# Raw data shape is (bands, lines, columns) — bands-first, unlike EMIT/NEON.
#
# NetCDF variable structure confirmed via EnSpec/hytools io/netcdf.py:
#   https://github.com/EnSpec/hytools/blob/master/hytools/io/netcdf.py
# ---------------------------------------------------------------------------


class AVIRIS3(Instrument):
    """
    AVIRIS-3 (Airborne Visible/InfraRed Imaging Spectrometer, 3rd generation).

    No bundled wavelength file exists for this instrument. Provide either a
    path to an AVIRIS-3 NetCDF4 product file or explicit wavelength arrays.

    Parameters
    ----------
    wavelengths : (n_bands,) array of band center wavelengths in nm
    fwhm : (n_bands,) array of FWHM values in nm. Defaults to 7.4 nm if not provided.
    nc_file : path to an AVIRIS-3 L1B or L2A NetCDF4 product file. If given,
              wavelengths and FWHM are read from the file.
    """

    def __init__(
        self,
        wavelengths: np.ndarray | None = None,
        fwhm: np.ndarray | None = None,
        nc_file: str | None = None,
    ) -> None:
        if nc_file is not None:
            self._wavelengths, self._fwhm = _read_spectral_from_nc(nc_file)
        elif wavelengths is not None:
            self._wavelengths = np.asarray(wavelengths, dtype=float)
            self._fwhm = (
                np.asarray(fwhm, dtype=float)
                if fwhm is not None
                else np.full(len(self._wavelengths), 7.4)
            )
        else:
            raise ValueError("Provide either 'nc_file' or 'wavelengths'.")

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    @property
    def fwhm(self) -> np.ndarray:
        return self._fwhm


def _read_spectral_from_nc(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read wavelengths and FWHM from an AVIRIS-3 NetCDF4 product file.

    The top-level group is 'radiance' (L1B) or 'reflectance' (L2A).
    Within that group, 'wavelength' and 'fwhm' are 1-D variables in nm.
    """
    try:
        import h5py  # type: ignore
    except ImportError as e:
        raise ImportError(
            "h5py is required to read AVIRIS-3 NetCDF files: pip install h5py"
        ) from e

    with h5py.File(filepath, "r") as f:
        group_name = _data_group(f)
        grp = f[group_name]
        wl = grp["wavelength"][()].astype(float)
        fwhm = grp["fwhm"][()].astype(float)

    # Ensure ascending wavelength order
    order = np.argsort(wl)
    return wl[order], fwhm[order]


def _data_group(f) -> str:
    """Return the name of the top-level data group ('radiance' or 'reflectance')."""
    for name in ("radiance", "reflectance"):
        if name in f:
            return name
    raise ValueError(
        f"Expected a top-level group named 'radiance' or 'reflectance' in the "
        f"AVIRIS-3 NetCDF file. Found: {list(f.keys())}"
    )
