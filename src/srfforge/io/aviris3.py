from pathlib import Path
import numpy as np
from ..instruments.aviris3 import AVIRIS3, _data_group

# AVIRIS-3 NetCDF4 file structure
# ---------------------------------------------------------------------------
# Product files from ORNL DAAC (doi:10.3334/ORNLDAAC/2356 / 2357) are
# NetCDF4 (HDF5-backed) with the following layout:
#
#   /radiance/                     ← top-level group (L1B); use 'reflectance' for L2A
#     radiance    (bands, lines, cols)  ← the actual data cube, bands-first
#     wavelength  (bands,)              ← center wavelength per band, nm
#     fwhm        (bands,)              ← FWHM per band, nm
#
# _FillValue attribute on the data variable marks no-data pixels.
# Reflectance (L2A) is stored as float32 in 0–1 range; no scale factor needed.
# Radiance (L1B) is in µW/(cm²·sr·nm).
#
# Structure confirmed via EnSpec/hytools io/netcdf.py:
#   https://github.com/EnSpec/hytools/blob/master/hytools/io/netcdf.py
# ---------------------------------------------------------------------------


def read_aviris3_nc(filepath: str | Path) -> tuple[np.ndarray, AVIRIS3, dict]:
    """
    Read an AVIRIS-3 L1B or L2A NetCDF4 product file.

    Parameters
    ----------
    filepath : path to the AVIRIS-3 .nc file

    Returns
    -------
    data : np.ndarray, shape (lines, cols, bands), float32
        For L2A reflectance files: values in 0–1 range.
        For L1B radiance files: values in µW/(cm²·sr·nm).
    aviris3 : AVIRIS3
        Instrument object populated with wavelengths and FWHM from the file.
    metadata : dict
        Extracted metadata: data_type ('radiance' or 'reflectance'),
        no_data fill value, and wavelength units.
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

        data_var = grp[group_name]  # nested variable with same name as group
        raw = data_var[()]  # (bands, lines, cols)

        no_data = data_var.attrs.get("_FillValue", None)
        if no_data is not None:
            no_data = float(np.asarray(no_data).flat[0])

        wl_units = "nm"
        if "units" in grp["wavelength"].attrs:
            wl_units = grp["wavelength"].attrs["units"]
            if isinstance(wl_units, bytes):
                wl_units = wl_units.decode()

    # Transpose from (bands, lines, cols) → (lines, cols, bands)
    data = np.moveaxis(raw, 0, -1).astype(np.float32)

    # Sort bands by ascending wavelength (should already be ascending, but be safe)
    order = np.argsort(wl)
    wl = wl[order]
    fwhm = fwhm[order]
    data = data[..., order]

    aviris3 = AVIRIS3(wavelengths=wl, fwhm=fwhm)
    metadata = {
        "data_type": group_name,
        "no_data": no_data,
        "wavelength_units": wl_units,
    }
    return data, aviris3, metadata
