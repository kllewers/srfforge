from pathlib import Path
import numpy as np
from .base import Instrument

_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_WL_FILE = _DATA_DIR / "EMIT_Wavelengths_20250721.txt"

# Bundled wavelength/FWHM source
# ---------------------------------------------------------------------------
# The file data/EMIT_Wavelengths_20250721.txt is the official EMIT spectral
# calibration file published by the EMIT Science Data System team at:
#   https://github.com/emit-sds/emit-sds-l1b/blob/main/data/emit/EMIT_Wavelengths_20250721.txt
#
# Format: 3 columns — band_index  wavelength_µm  fwhm_µm
# Bands with fwhm == 0 are masked (detector edge / bad channels); we drop them.
# After filtering: 288 good bands, 366–2500 nm, FWHM ~8.4–8.8 nm.
#
# EMIT NetCDF4 product files (L1B/L2A from LP DAAC) also embed per-acquisition
# wavelength and FWHM in the group  /sensor_band_parameters/wavelengths  and
# /sensor_band_parameters/fwhm  (units: nm).  Pass a .nc file path to use those
# instead of the bundled nominal values.
# ---------------------------------------------------------------------------


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
    """Parse emit-sds-l1b wavelength file: columns are band_idx, wl_um, fwhm_um.
    Source: https://github.com/emit-sds/emit-sds-l1b/tree/main/data/emit
    """
    data = np.loadtxt(path)
    wl_nm = data[:, 1] * 1000.0
    fwhm_nm = data[:, 2] * 1000.0
    mask = fwhm_nm > 0
    wl_nm, fwhm_nm = wl_nm[mask], fwhm_nm[mask]
    order = np.argsort(wl_nm)
    return wl_nm[order], fwhm_nm[order]


def _load_wavelengths_netcdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract wavelengths and FWHM from an EMIT NetCDF4 product file.

    EMIT L1B/L2A products from LP DAAC store spectral parameters in the
    group /sensor_band_parameters with variables 'wavelengths' and 'fwhm'
    (both in nm).  Confirmed via EnSpec/hytools netcdf.py and the EMIT
    L2A user guide: https://lpdaac.usgs.gov/documents/1569/EMITL2ARFL_User_Guide_v1.pdf

    Uses h5py rather than netCDF4 — NetCDF4 files are HDF5-backed, so h5py
    reads them directly without an extra dependency.
    """
    try:
        import h5py  # type: ignore
    except ImportError as e:
        raise ImportError(
            "h5py is required to read EMIT product files: pip install h5py"
        ) from e

    with h5py.File(path, "r") as f:
        spb = f.get("sensor_band_parameters")
        if spb is None:
            raise ValueError(
                f"No 'sensor_band_parameters' group found in {path}. "
                "Expected a standard EMIT L2A reflectance product."
            )
        wl_nm = spb["wavelengths"][()].astype(float)
        fwhm_nm = spb["fwhm"][()].astype(float)

    order = np.argsort(wl_nm)
    return wl_nm[order], fwhm_nm[order]
