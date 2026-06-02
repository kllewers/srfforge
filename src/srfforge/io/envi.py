from pathlib import Path
import numpy as np
from ..instruments.neon import NEON


def read_neon_envi(hdr_file: str | Path) -> tuple[np.ndarray, NEON, dict]:
    """
    Read a NEON AOP reflectance ENVI file.

    Parameters
    ----------
    hdr_file : path to the ENVI .hdr file (the binary data file is inferred from it)

    Returns
    -------
    reflectance : np.ndarray, shape (rows, cols, n_bands), float32
        Surface reflectance values scaled to 0–1.
    neon : NEON
        Instrument object populated with wavelengths and FWHM from the header.
    metadata : dict
        Raw header metadata dict from spectral plus parsed 'scale_factor'.
    """
    try:
        import spectral  # type: ignore
    except ImportError as e:
        raise ImportError(
            "spectral is required to read ENVI files: pip install spectral"
        ) from e

    img = spectral.open_image(str(hdr_file))
    meta = dict(img.metadata)

    wl = np.array([float(w) for w in meta["wavelength"]], dtype=float)

    if "fwhm" in meta:
        fwhm = np.array([float(f) for f in meta["fwhm"]], dtype=float)
    else:
        fwhm = np.full(len(wl), 5.0)

    # Wavelength units: convert µm → nm if needed
    wl_units = meta.get("wavelength units", "").lower()
    if wl_units in ("micrometers", "um", "µm") or (wl_units == "" and wl.max() < 10):
        wl = wl * 1000.0
        fwhm = fwhm * 1000.0

    # Scale factor: NEON typically stores reflectance * 10000 as int16
    scale = _parse_scale(meta)

    data = img.load()  # (rows, cols, bands), original dtype
    refl = np.asarray(data, dtype=np.float32) / scale

    neon = NEON(wavelengths=wl, fwhm=fwhm)
    metadata = {**meta, "scale_factor": scale}
    return refl, neon, metadata


def _parse_scale(meta: dict) -> float:
    """Extract scale factor from ENVI header metadata."""
    for key in ("reflectance scale factor", "scale factor", "scalefactor"):
        if key in meta:
            val = float(meta[key])
            # A scale factor < 1 means multiply, not divide; convert to divisor
            return val if val >= 1.0 else 1.0 / val
    return 10000.0  # NEON AOP default
