# srfforge

Convolve hyperspectral reflectance data between remote sensing instruments using Gaussian spectral response functions (SRFs).

Currently supports convolving **NEON AOP** data to **EMIT** spectral sampling. The instrument framework is designed to be extended to other sensors.

## Installation

```bash
# Core (band convolution only)
pip install -e "."

# With EarthData search (find co-located NEON + EMIT granules)
pip install -e ".[search]"

# With plotting helpers
pip install -e ".[plot]"

# Everything
pip install -e ".[full]"

# Development
pip install -e ".[dev]"
```

**Core dependencies:** `numpy`, `h5py`
**Search extras:** `earthaccess`, `neonutilities`, `shapely`, `pyproj`
**Plot extras:** `matplotlib`

## Find co-located NEON + EMIT data and convolve

```python
from srfforge.search import find_overlapping
from srfforge.io import read_neon_h5
from srfforge import BandConvolver, EMIT
from srfforge.plot import compare_spectra, rgb_quicklook

# 1. Find overlapping granules (requires pip install 'srfforge[search]')
pairs = find_overlapping(site="NIWO", year=2023, max_cloud=30)
print(f"Found {len(pairs)} NEON tile / EMIT granule pairs")

# 2. Read NEON tile
refl, neon, meta = read_neon_h5("NEON_D13_NIWO_DP3_454000_4431000_reflectance.h5")

# 3. Convolve to EMIT spectral sampling
emit_refl = BandConvolver(source=neon, target=EMIT())(refl)

# 4. Plot (requires pip install 'srfforge[plot]')
compare_spectra(refl, emit_refl, source=neon, target=EMIT())
rgb_quicklook(refl, instrument=neon)
```

## Quick start

### File-based (NEON HDF5 → EMIT)

```python
from srfforge import BandConvolver, EMIT
from srfforge.io import read_neon_h5

# Read a NEON AOP reflectance tile
refl, neon, meta = read_neon_h5("NEON_D02_HARV_DP3_726000_4700000_reflectance.h5")
# refl: (rows, cols, n_neon_bands), float32, scaled 0–1

# Convolve to EMIT spectral sampling
conv = BandConvolver(source=neon, target=EMIT())
emit_refl = conv(refl)  # (rows, cols, 288)
```

### Array-based

```python
import numpy as np
from srfforge import BandConvolver, NEON, EMIT

neon = NEON(wavelengths=my_wavelengths_nm, fwhm=my_fwhm_nm)
emit = EMIT()

conv = BandConvolver(source=neon, target=emit)
emit_refl = conv(neon_refl)  # (..., n_emit_bands)
```

### Use your own EMIT product file

```python
from srfforge.instruments import EMIT

# Load band parameters from an actual EMIT L2A reflectance product
emit = EMIT(srf_file="EMIT_L2A_RFL_001_20220903T163129_2224611_012.nc")
```

## How it works

For each EMIT band with center wavelength $\lambda_i$ and FWHM $w_i$, a Gaussian SRF is evaluated at every NEON band center:

$$M_{ij} = \exp\!\left(-\frac{(\lambda_j^{\text{NEON}} - \lambda_i^{\text{EMIT}})^2}{2\sigma_i^2}\right), \quad \sigma_i = \frac{w_i}{2\sqrt{2\ln 2}}$$

Each row is normalized so weights sum to 1. The convolution is then a matrix multiply:

$$R^{\text{EMIT}} = M \cdot R^{\text{NEON}}$$

## EMIT band data

The bundled EMIT spectral calibration data (`EMIT_Wavelengths_20250721.txt`) is sourced from the official [`emit-sds/emit-sds-l1b`](https://github.com/emit-sds/emit-sds-l1b) repository. It contains **288 good bands** covering **366–2500 nm** with FWHM of ~8.4–8.8 nm.

EMIT acquisition data (L1B/L2A NetCDF4 products) can be downloaded from [LP DAAC](https://lpdaac.usgs.gov/products/emitl2arfl/) via NASA Earthdata.

## NEON data

NEON AOP reflectance data (HDF5 format) is available through the [NEON Data Portal](https://data.neonscience.org/). The relevant data product is **DP3.30006.001** (Spectrometer orthorectified surface directional reflectance).

The HDF5 files contain per-tile wavelength centers and FWHM, which `read_neon_h5` reads automatically to build the source instrument.

## Running tests

```bash
pytest
```

## Project structure

```
src/srfforge/
├── __init__.py          # public API: BandConvolver, NEON, EMIT, AVIRIS3
├── convolve.py          # BandConvolver class
├── srf.py               # Gaussian SRF math (build_convolution_matrix)
├── instruments/
│   ├── __init__.py      # re-exports NEON, EMIT, AVIRIS3
│   ├── base.py          # abstract Instrument base class
│   ├── emit.py          # EMIT instrument (bundled or file-based wavelengths)
│   ├── neon.py          # NEON instrument (array or HDF5 file)
│   └── aviris3.py       # AVIRIS-3 instrument (array or NetCDF4 file)
├── io/
│   ├── __init__.py      # re-exports read_neon_h5, read_neon_envi, read_aviris3_nc
│   ├── hdf5.py          # read_neon_h5   — reads NEON .h5 reflectance tile + spatial metadata
│   ├── envi.py          # read_neon_envi — reads NEON ENVI .hdr/.bin file
│   └── aviris3.py       # read_aviris3_nc — reads AVIRIS-3 L1B/L2A .nc file
├── search/              # [search] extra — EarthData colocation
│   ├── __init__.py      # find_overlapping(site, year) → list[OverlapResult]
│   ├── _emit.py         # search_emit_granules via earthaccess
│   └── _neon.py         # get_neon_tile_bounds via neonutilities + pyproj
├── plot/                # [plot] extra — visualization helpers
│   ├── __init__.py      # compare_spectra, rgb_quicklook, plot_residuals
│   └── core.py          # matplotlib implementations
└── data/
    └── EMIT_Wavelengths_20250721.txt   # bundled EMIT spectral calibration
```

## Module call graph

```
User code
    │
    ▼
srfforge/__init__.py  ──exports──►  BandConvolver, NEON, EMIT, AVIRIS3
srfforge/io/__init__.py  ────────►  read_neon_h5, read_neon_envi, read_aviris3_nc

─────────────────────────────────────────────────────────────────────────────
Instrument hierarchy
─────────────────────────────────────────────────────────────────────────────

             instruments/base.py
                  Instrument  (abstract: .wavelengths, .fwhm)
                  ▲     ▲     ▲
                  │     │     │
            neon.py  emit.py  aviris3.py
              NEON    EMIT    AVIRIS3

─────────────────────────────────────────────────────────────────────────────
Instrument initialization
─────────────────────────────────────────────────────────────────────────────

EMIT(srf_file=None)
  ├─ srf_file=None  ──► _load_wavelengths_txt(bundled .txt)
  ├─ srf_file=".nc" ──► _load_wavelengths_netcdf()  [h5py]
  └─ srf_file=".txt"──► _load_wavelengths_txt()

NEON(wavelengths=..., fwhm=...)     ← arrays passed directly
NEON(h5_file="...")  ──────────────► _read_spectral_from_h5()  [h5py]

AVIRIS3(wavelengths=..., fwhm=...)  ← arrays passed directly
AVIRIS3(nc_file="...")  ───────────► _read_spectral_from_nc()
                                          └──► _data_group()
                                               (detects 'radiance' | 'reflectance')

─────────────────────────────────────────────────────────────────────────────
Convolution
─────────────────────────────────────────────────────────────────────────────

BandConvolver(source, target)
  │
  ├─ __init__  ──► srf.py::build_convolution_matrix(
  │                    source.wavelengths, target.wavelengths, target.fwhm
  │                )  →  M  shape (n_target, n_source)
  │
  └─ __call__(reflectance)  ──►  reflectance @ M.T
                                 shape (..., n_target_bands)

─────────────────────────────────────────────────────────────────────────────
IO readers  (file → array + Instrument + metadata)
─────────────────────────────────────────────────────────────────────────────

io/hdf5.py::read_neon_h5(path)
  │  h5py reads:  /{site}/Reflectance/Metadata/Spectral_Data/{Wavelength,FWHM}
  │               /{site}/Reflectance/Reflectance_Data  (rows×cols×bands)
  └─ returns ──►  (reflectance ndarray, NEON(wavelengths, fwhm), metadata dict)

io/envi.py::read_neon_envi(hdr_file)
  │  spectral reads: ENVI .hdr + binary sidecar
  └─ returns ──►  (reflectance ndarray, NEON(wavelengths, fwhm), metadata dict)

io/aviris3.py::read_aviris3_nc(path)
  │  h5py reads:  /{radiance|reflectance}/{wavelength,fwhm,data_cube}
  │               calls instruments/aviris3.py::_data_group()
  └─ returns ──►  (data ndarray, AVIRIS3(wavelengths, fwhm), metadata dict)
                  data transposed: (bands,lines,cols) → (lines,cols,bands)
```
