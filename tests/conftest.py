import numpy as np
import pytest


@pytest.fixture
def neon_wavelengths():
    return np.linspace(381.0, 2511.0, 426)


@pytest.fixture
def neon_fwhm(neon_wavelengths):
    return np.full(len(neon_wavelengths), 5.0)


@pytest.fixture
def synthetic_spectrum(neon_wavelengths):
    """Smooth synthetic spectrum: mix of two Gaussians to simulate vegetation."""
    rng = np.random.default_rng(42)
    base = 0.05 + 0.45 * np.exp(-0.5 * ((neon_wavelengths - 800) / 200) ** 2)
    noise = rng.normal(0, 0.001, len(neon_wavelengths))
    return np.clip(base + noise, 0, 1).astype(np.float32)


@pytest.fixture
def neon_h5_file(tmp_path, neon_wavelengths, neon_fwhm):
    """Minimal NEON AOP HDF5 file with a 10×10 synthetic scene."""
    import h5py

    filepath = tmp_path / "HARV_test_reflectance.h5"
    rng = np.random.default_rng(0)
    refl_int = (rng.uniform(0, 0.8, (10, 10, 426)) * 10000).astype(np.int16)

    with h5py.File(filepath, "w") as f:
        site = f.create_group("HARV")
        ref = site.create_group("Reflectance")
        meta = ref.create_group("Metadata")
        spec = meta.create_group("Spectral_Data")
        spec.create_dataset("Wavelength", data=neon_wavelengths)
        spec.create_dataset("FWHM", data=neon_fwhm)
        ds = ref.create_dataset("Reflectance_Data", data=refl_int)
        ds.attrs["Scale_Factor"] = 10000.0

    return filepath


@pytest.fixture
def neon_envi_file(tmp_path, neon_wavelengths, neon_fwhm):
    """Minimal NEON ENVI file (BIP int16) for testing."""
    n_rows, n_cols, n_bands = 5, 5, len(neon_wavelengths)
    rng = np.random.default_rng(1)
    data = (rng.uniform(0, 0.8, (n_rows, n_cols, n_bands)) * 10000).astype(np.int16)

    bin_path = tmp_path / "neon_test"
    hdr_path = tmp_path / "neon_test.hdr"

    # Write binary (BIP: pixels interleaved by band)
    data.tofile(str(bin_path))

    wl_str = ", ".join(f"{w:.4f}" for w in neon_wavelengths)
    fwhm_str = ", ".join(f"{f:.4f}" for f in neon_fwhm)
    hdr_path.write_text(
        f"ENVI\n"
        f"samples = {n_cols}\n"
        f"lines = {n_rows}\n"
        f"bands = {n_bands}\n"
        f"data type = 2\n"
        f"interleave = bip\n"
        f"byte order = 0\n"
        f"wavelength units = Nanometers\n"
        f"reflectance scale factor = 10000\n"
        f"data ignore value = -9999\n"
        f"wavelength = {{{wl_str}}}\n"
        f"fwhm = {{{fwhm_str}}}\n"
    )
    return hdr_path
