# srfforge

Convolve hyperspectral reflectance data between remote sensing instruments using Gaussian spectral response functions (SRFs).

Currently supports convolving **NEON AOP** data to **EMIT** spectral sampling. The instrument framework is designed to be extended to other sensors.

## Installation

```bash
pip install -e ".[dev]"
```

**Dependencies:** `numpy`, `scipy`, `h5py`

To read EMIT NetCDF4 product files directly, also install `netCDF4`:
```bash
pip install netCDF4
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
├── convolve.py          # BandConvolver
├── srf.py               # Gaussian SRF math
├── instruments/
│   ├── base.py          # abstract Instrument
│   ├── emit.py          # EMIT
│   └── neon.py          # NEON
├── io/
│   └── hdf5.py          # read_neon_h5
└── data/
    └── EMIT_Wavelengths_20250721.txt
```
