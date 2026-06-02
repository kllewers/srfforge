from pathlib import Path
import numpy as np
from ..instruments.neon import NEON

# ─────────────────────────────────────────────────────────────────────────────
# Module: io/hdf5.py  —  NEON HDF5 file reader
#
#   called by ◄── user code
#   calls     ──► instruments/neon.py::NEON  (to build instrument object)
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
        Extracted metadata: site code, scale factor, and bad band windows if present.
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
        refl = refl_ds[:].astype(np.float32) / scale

        metadata: dict = {"site": site, "scale_factor": scale}
        for key in ("Bad_Band_Window1", "Bad_Band_Window2"):
            if key in spec:
                metadata[key.lower()] = spec[key][:]

    neon = NEON(wavelengths=wl, fwhm=fwhm)
    return refl, neon, metadata
