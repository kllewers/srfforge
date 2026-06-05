from pathlib import Path
import numpy as np
from ..instruments.neon import NEON

# ─────────────────────────────────────────────────────────────────────────────
# Module: io/hdf5.py  —  NEON HDF5 file reader
#
#   called by ◄── user code
#   calls     ──► instruments/neon.py::NEON  (to build instrument object)
#                 _parse_map_info  (spatial metadata helper)
# ─────────────────────────────────────────────────────────────────────────────


#--------------------------------------
# Called by: user code
# Calls:     NEON (instruments/neon.py)
#--------------------------------------
def read_neon_h5(filepath: str | Path) -> tuple[np.ndarray, NEON, dict]:
    """
    Read a NEON AOP reflectance HDF5 file.

    Parameters
    ----------
    filepath : path to the NEON .h5 file

    Returns
    -------
    reflectance : np.ndarray, shape (rows, cols, n_bands), float32
        Surface reflectance values scaled to 0–1.
    neon : NEON
        Instrument object populated with wavelengths and FWHM from the file.
    metadata : dict
        Extracted metadata including:
        - site (str): NEON site code
        - scale_factor (float)
        - bad_band_window1/2 (array, if present)
        - epsg (int, if present): coordinate reference system EPSG code
        - utm_bounds (dict, if present): x_min/y_min/x_max/y_max in UTM metres,
          utm_zone, hemisphere — useful for spatial overlap checks
    """
    try:
        import h5py  # type: ignore
    except ImportError as e:
        raise ImportError("h5py is required to read NEON HDF5 files: pip install h5py") from e

    with h5py.File(filepath, "r") as f:
        site = next(iter(f.keys()))
        ref_grp = f[site]["Reflectance"]

        spec = ref_grp["Metadata"]["Spectral_Data"]
        wl = spec["Wavelength"][:].astype(float)
        fwhm = spec["FWHM"][:].astype(float)

        refl_ds = ref_grp["Reflectance_Data"]
        scale = float(refl_ds.attrs.get("Scale_Factor", 10000.0))

        # Mask fill value before scaling so integer comparison is exact.
        # Attribute name varies across NEON product versions; -9999 is always the value.
        raw = refl_ds[:]
        fill_val = None
        for attr in ("Data_Ignore_Value", "_FillValue", "fill_value", "missing_value"):
            v = refl_ds.attrs.get(attr, None)
            if v is not None:
                fill_val = int(np.asarray(v).flat[0])
                break
        if fill_val is None:
            fill_val = -9999  # NEON universal default

        refl = raw.astype(np.float32) / scale
        refl[raw == fill_val] = np.nan

        metadata: dict = {"site": site, "scale_factor": scale}
        for key in ("Bad_Band_Window1", "Bad_Band_Window2"):
            if key in spec:
                metadata[key.lower()] = spec[key][:]

        # Spatial metadata from coordinate system group
        coord_grp = ref_grp["Metadata"].get("Coordinate_System")
        if coord_grp is not None:
            if "EPSG Code" in coord_grp:
                raw = coord_grp["EPSG Code"][()]
                try:
                    metadata["epsg"] = int(raw.decode() if isinstance(raw, bytes) else raw)
                except (ValueError, AttributeError):
                    pass
            if "Map_Info" in coord_grp:
                map_info = _parse_map_info(coord_grp["Map_Info"][()])
                rows_n, cols_n = refl_ds.shape[:2]
                metadata["utm_bounds"] = {
                    "x_min": map_info["x_ul"],
                    "y_max": map_info["y_ul"],
                    "x_max": map_info["x_ul"] + cols_n * map_info["x_res"],
                    "y_min": map_info["y_ul"] - rows_n * map_info["y_res"],
                    "utm_zone": map_info.get("utm_zone"),
                    "hemisphere": map_info.get("hemisphere", "north"),
                }

    neon = NEON(wavelengths=wl, fwhm=fwhm)
    return refl, neon, metadata


#--------------------------------------
# Called by: read_neon_h5
# Calls:     none
#--------------------------------------
def _parse_map_info(raw) -> dict:
    """
    Parse an ENVI-style Map_Info string from a NEON HDF5 Coordinate_System group.

    Fields (comma-separated): projection, start_col, start_row, x_ul, y_ul,
    x_res, y_res, utm_zone, hemisphere, datum, ...
    """
    s = raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)
    parts = [p.strip() for p in s.split(",")]
    info: dict = {}
    try:
        info["projection"] = parts[0]
        info["x_ul"] = float(parts[3])
        info["y_ul"] = float(parts[4])
        info["x_res"] = float(parts[5])
        info["y_res"] = abs(float(parts[6]))  # stored negative in some files
        if len(parts) > 7:
            info["utm_zone"] = int(parts[7])
        if len(parts) > 8:
            info["hemisphere"] = "south" if "south" in parts[8].lower() else "north"
    except (IndexError, ValueError):
        pass
    return info
