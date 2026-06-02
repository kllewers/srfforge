"""Tests for AVIRIS-3 instrument and NetCDF IO."""
import numpy as np
import pytest
import h5py
from srfforge import BandConvolver, EMIT, AVIRIS3
from srfforge.io import read_aviris3_nc


@pytest.fixture
def aviris3_wavelengths():
    return np.linspace(390.0, 2500.0, 285)


@pytest.fixture
def aviris3_fwhm(aviris3_wavelengths):
    return np.full(len(aviris3_wavelengths), 7.4)


@pytest.fixture
def aviris3_nc_file(tmp_path, aviris3_wavelengths, aviris3_fwhm):
    """Minimal AVIRIS-3 L2A NetCDF4 fixture matching the real file structure.

    Structure mirrors what hytools/io/netcdf.py expects for sensor='AV':
      /reflectance/
        reflectance  (bands, lines, cols)
        wavelength   (bands,)
        fwhm         (bands,)
    """
    filepath = tmp_path / "AV3_L2A_RFL_test.nc"
    n_bands, n_lines, n_cols = len(aviris3_wavelengths), 8, 8
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 0.8, (n_bands, n_lines, n_cols)).astype(np.float32)

    with h5py.File(filepath, "w") as f:
        grp = f.create_group("reflectance")
        ds = grp.create_dataset("reflectance", data=data)
        ds.attrs["_FillValue"] = np.float32(-9999.0)
        wl_ds = grp.create_dataset("wavelength", data=aviris3_wavelengths)
        wl_ds.attrs["units"] = "nm"
        grp.create_dataset("fwhm", data=aviris3_fwhm)

    return filepath


@pytest.fixture
def aviris3_nc_radiance_file(tmp_path, aviris3_wavelengths, aviris3_fwhm):
    """AVIRIS-3 L1B radiance NetCDF4 fixture (group named 'radiance')."""
    filepath = tmp_path / "AV3_L1B_RDN_test.nc"
    n_bands, n_lines, n_cols = len(aviris3_wavelengths), 4, 4
    rng = np.random.default_rng(7)
    data = rng.uniform(0, 500, (n_bands, n_lines, n_cols)).astype(np.float32)

    with h5py.File(filepath, "w") as f:
        grp = f.create_group("radiance")
        ds = grp.create_dataset("radiance", data=data)
        ds.attrs["_FillValue"] = np.float32(-9999.0)
        grp.create_dataset("wavelength", data=aviris3_wavelengths)
        grp.create_dataset("fwhm", data=aviris3_fwhm)

    return filepath


class TestAVIRIS3Instrument:
    def test_array_init(self, aviris3_wavelengths, aviris3_fwhm):
        av3 = AVIRIS3(wavelengths=aviris3_wavelengths, fwhm=aviris3_fwhm)
        assert len(av3.wavelengths) == 285
        np.testing.assert_array_equal(av3.fwhm, aviris3_fwhm)

    def test_default_fwhm(self, aviris3_wavelengths):
        av3 = AVIRIS3(wavelengths=aviris3_wavelengths)
        assert np.all(av3.fwhm == 7.4)

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            AVIRIS3()

    def test_nc_file_init(self, aviris3_nc_file, aviris3_wavelengths):
        av3 = AVIRIS3(nc_file=aviris3_nc_file)
        np.testing.assert_allclose(av3.wavelengths, aviris3_wavelengths, atol=0.01)

    def test_wavelengths_ascending(self, aviris3_nc_file):
        av3 = AVIRIS3(nc_file=aviris3_nc_file)
        assert np.all(np.diff(av3.wavelengths) > 0)

    def test_repr(self, aviris3_wavelengths):
        av3 = AVIRIS3(wavelengths=aviris3_wavelengths)
        assert "AVIRIS3" in repr(av3)
        assert "nm" in repr(av3)


class TestReadAVIRIS3NC:
    def test_output_shape_reflectance(self, aviris3_nc_file, aviris3_wavelengths):
        data, av3, meta = read_aviris3_nc(aviris3_nc_file)
        # Should be transposed to (lines, cols, bands)
        assert data.shape == (8, 8, len(aviris3_wavelengths))
        assert data.dtype == np.float32

    def test_output_shape_radiance(self, aviris3_nc_radiance_file, aviris3_wavelengths):
        data, av3, meta = read_aviris3_nc(aviris3_nc_radiance_file)
        assert data.shape == (4, 4, len(aviris3_wavelengths))

    def test_metadata_data_type(self, aviris3_nc_file):
        _, _, meta = read_aviris3_nc(aviris3_nc_file)
        assert meta["data_type"] == "reflectance"

    def test_metadata_data_type_radiance(self, aviris3_nc_radiance_file):
        _, _, meta = read_aviris3_nc(aviris3_nc_radiance_file)
        assert meta["data_type"] == "radiance"

    def test_no_data_value(self, aviris3_nc_file):
        _, _, meta = read_aviris3_nc(aviris3_nc_file)
        assert meta["no_data"] == pytest.approx(-9999.0)

    def test_wavelengths_match(self, aviris3_nc_file, aviris3_wavelengths):
        _, av3, _ = read_aviris3_nc(aviris3_nc_file)
        np.testing.assert_allclose(av3.wavelengths, aviris3_wavelengths, atol=0.01)

    def test_unknown_group_raises(self, tmp_path):
        bad_file = tmp_path / "bad.nc"
        with h5py.File(bad_file, "w") as f:
            f.create_group("unknown_group")
        with pytest.raises(ValueError, match="radiance.*reflectance"):
            read_aviris3_nc(bad_file)


class TestAVIRIS3Pipeline:
    def test_convolve_to_emit(self, aviris3_nc_file):
        data, av3, _ = read_aviris3_nc(aviris3_nc_file)
        emit = EMIT()
        conv = BandConvolver(source=av3, target=emit)
        out = conv(data)
        assert out.shape == (8, 8, len(emit.wavelengths))

    def test_reflectance_range(self, aviris3_nc_file):
        data, av3, _ = read_aviris3_nc(aviris3_nc_file)
        conv = BandConvolver(source=av3, target=EMIT())
        out = conv(data)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
