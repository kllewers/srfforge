import re
import numpy as np
from pathlib import Path
from .base import Instrument

# ─────────────────────────────────────────────────────────────────────────────
# Module: instruments/gliht.py  —  G-LiHT Headwall Micro-Hyperspec VNIR
#
#   called by ◄── user code / BandConvolver (via .wavelengths / .fwhm)
#   calls     ──► _load_from_hdr  (local, parses ENVI .hdr)
# ─────────────────────────────────────────────────────────────────────────────

# Nominal spectral sampling for the Headwall Micro-Hyperspec E-Series
# as flown on G-LiHT v.2 (2017+): 400–1000 nm, 1.6 nm/band, 5 nm FWHM.
# Source: gliht.gsfc.nasa.gov/index.php?section=50
_NOMINAL_WL   = np.arange(400.0, 1001.7, 1.6)   # ~376 bands
_NOMINAL_FWHM = 5.0                               # nm


#--------------------------------------
# Called by: user code
# Calls:     _load_from_hdr (if hdr_file given)
#--------------------------------------
class GLiHT(Instrument):
    """
    G-LiHT (Goddard's LiDAR, Hyperspectral & Thermal) Headwall Micro-Hyperspec VNIR.

    Spectral range: 400–1000 nm, ~1.6 nm/band sampling, 5 nm FWHM.

    Parameters
    ----------
    hdr_file : path to ENVI .hdr from a G-LiHT data download (preferred —
               uses the per-acquisition calibration wavelengths).
    wavelengths : explicit wavelength array in nm. Takes priority over hdr_file.
    fwhm : explicit FWHM array or scalar in nm. Used with explicit wavelengths,
           or as a fallback when the .hdr has no 'fwhm' field.
    fwhm_nm : scalar FWHM (nm) when no fwhm source is available. Default 5.0.
    """

    def __init__(
        self,
        hdr_file: str | Path | None = None,
        wavelengths: np.ndarray | None = None,
        fwhm: np.ndarray | float | None = None,
        fwhm_nm: float = 5.0,
    ) -> None:
        if wavelengths is not None:
            wl = np.asarray(wavelengths, dtype=float)
            if fwhm is not None:
                fw = np.broadcast_to(np.asarray(fwhm, dtype=float), wl.shape).copy()
            else:
                fw = np.full(len(wl), float(fwhm_nm))
        elif hdr_file is not None:
            wl, fw = _load_from_hdr(Path(hdr_file), fallback_fwhm=fwhm_nm)
        else:
            wl = _NOMINAL_WL.copy()
            fw = np.full(len(wl), float(fwhm_nm))

        self._wavelengths = wl
        self._fwhm = fw

    #--------------------------------------
    # Called by: BandConvolver.__init__ (convolve.py)
    # Calls:     none
    #--------------------------------------
    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    #--------------------------------------
    # Called by: BandConvolver.__init__ (convolve.py)
    # Calls:     none
    #--------------------------------------
    @property
    def fwhm(self) -> np.ndarray:
        return self._fwhm


#--------------------------------------
# Called by: GLiHT.__init__
# Calls:     none
#--------------------------------------
def _load_from_hdr(hdr_path: Path, fallback_fwhm: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse an ENVI .hdr file and return (wavelengths_nm, fwhm_nm).

    Handles both µm and nm units, and multi-line brace-delimited lists.
    No external dependencies — parses the plain-text header directly.
    """
    text = hdr_path.read_text(errors="replace")

    def _extract_list(key: str) -> list[float] | None:
        # Match 'key = { ... }' across multiple lines
        pattern = rf"(?i){re.escape(key)}\s*=\s*\{{([^}}]+)\}}"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return [float(v) for v in re.split(r"[,\n\r]+", m.group(1)) if v.strip()]
        # Match 'key = value' (single value, no braces)
        pattern2 = rf"(?i)^{re.escape(key)}\s*=\s*(.+)$"
        m2 = re.search(pattern2, text, re.MULTILINE)
        if m2:
            return [float(m2.group(1).strip())]
        return None

    wl_raw = _extract_list("wavelength")
    if not wl_raw:
        raise ValueError(f"No 'wavelength' field found in {hdr_path}")
    wl = np.array(wl_raw, dtype=float)

    fwhm_raw = _extract_list("fwhm")
    fw = np.array(fwhm_raw, dtype=float) if fwhm_raw else np.full(len(wl), fallback_fwhm)

    # Convert µm → nm if needed
    wl_units = ""
    m_units = re.search(r"(?i)wavelength\s+units\s*=\s*(.+)", text)
    if m_units:
        wl_units = m_units.group(1).strip().lower()
    if wl_units in ("micrometers", "um", "µm") or (wl_units == "" and wl.max() < 10):
        wl *= 1000.0
        fw *= 1000.0

    order = np.argsort(wl)
    return wl[order], fw[order]
