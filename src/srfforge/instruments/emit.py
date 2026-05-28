from pathlib import Path
import numpy as np
from .base import Instrument

_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_WL_FILE = _DATA_DIR / "EMIT_Wavelengths_20250721.txt"


class EMIT(Instrument):
    """
    EMIT (Earth Surface Mineral Dust Source Investigation) instrument.

    Parameters
    ----------
    srf_file : optional path to a wavelengths file. Accepts:
        - emit-sds-l1b format .txt (3 columns: band_idx, wavelength_um, fwhm_um)
        - EMIT NetCDF4 product file (.nc/.nc4) with sensor_band_parameters group
        If None, uses the bundled nominal wavelengths from emit-sds-l1b (2025-07-21).
    """

    def __init__(self, srf_file: str | None = None) -> None:
        if srf_file is None:
            self._wavelengths, self._fwhm = _load_wavelengths_txt(_DEFAULT_WL_FILE)
        else:
            path = Path(srf_file)
            if path.suffix in (".nc", ".nc4"):
                self._wavelengths, self._fwhm = _load_wavelengths_netcdf(path)
            else:
                self._wavelengths, self._fwhm = _load_wavelengths_txt(path)

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    @property
    def fwhm(self) -> np.ndarray:
        return self._fwhm


def _load_wavelengths_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse emit-sds-l1b wavelength file: columns are band_idx, wl_um, fwhm_um."""
    data = np.loadtxt(path)
    wl_nm = data[:, 1] * 1000.0
    fwhm_nm = data[:, 2] * 1000.0
    mask = fwhm_nm > 0
    wl_nm, fwhm_nm = wl_nm[mask], fwhm_nm[mask]
    order = np.argsort(wl_nm)
    return wl_nm[order], fwhm_nm[order]


def _load_wavelengths_netcdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract wavelengths and FWHM from an EMIT NetCDF4 product file."""
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError as e:
        raise ImportError(
            "netCDF4 is required to read EMIT product files: pip install netCDF4"
        ) from e

    with nc.Dataset(path) as ds:
        spb = ds.groups.get("sensor_band_parameters")
        if spb is None:
            raise ValueError(
                f"No 'sensor_band_parameters' group found in {path}. "
                "Expected a standard EMIT L2A reflectance product."
            )
        wl_nm = np.array(spb.variables["wavelengths"][:], dtype=float)
        fwhm_nm = np.array(spb.variables["fwhm"][:], dtype=float)

    order = np.argsort(wl_nm)
    return wl_nm[order], fwhm_nm[order]
