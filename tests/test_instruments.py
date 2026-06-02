import numpy as np
import pytest
from srfforge.instruments import NEON, EMIT


class TestEMIT:
    def test_default_loads(self):
        emit = EMIT()
        assert len(emit.wavelengths) == len(emit.fwhm)
        assert len(emit.wavelengths) > 0

    def test_wavelengths_ascending(self):
        emit = EMIT()
        assert np.all(np.diff(emit.wavelengths) > 0)

    def test_wavelength_range(self):
        emit = EMIT()
        assert emit.wavelengths[0] > 300
        assert emit.wavelengths[-1] < 2600

    def test_fwhm_positive(self):
        emit = EMIT()
        assert np.all(emit.fwhm > 0)

    def test_repr(self):
        emit = EMIT()
        r = repr(emit)
        assert "EMIT" in r
        assert "nm" in r

    def test_custom_txt_file(self, tmp_path):
        # Write a minimal 3-column wavelengths file (same format as emit-sds-l1b)
        txt = tmp_path / "wl.txt"
        rows = [(i, 0.5 + i * 0.01, 0.008) for i in range(10)]
        txt.write_text("\n".join(f"{i} {w} {f}" for i, w, f in rows))
        emit = EMIT(srf_file=str(txt))
        assert len(emit.wavelengths) == 10

    def test_netcdf_missing_group_raises(self, tmp_path):
        # Reader now uses h5py (no netCDF4 needed). A valid HDF5 file that lacks
        # the sensor_band_parameters group should raise ValueError.
        import h5py
        nc_file = tmp_path / "fake.nc"
        with h5py.File(nc_file, "w") as f:
            f.create_group("some_other_group")
        with pytest.raises(ValueError, match="sensor_band_parameters"):
            EMIT(srf_file=str(nc_file))


class TestNEON:
    def test_array_init(self):
        wl = np.linspace(381, 2511, 426)
        neon = NEON(wavelengths=wl)
        assert len(neon.wavelengths) == 426
        assert np.all(neon.fwhm == 5.0)

    def test_array_init_with_fwhm(self):
        wl = np.linspace(381, 2511, 426)
        fwhm = np.full(426, 5.5)
        neon = NEON(wavelengths=wl, fwhm=fwhm)
        np.testing.assert_array_equal(neon.fwhm, fwhm)

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            NEON()

    def test_repr(self):
        wl = np.linspace(381, 2511, 426)
        neon = NEON(wavelengths=wl)
        assert "NEON" in repr(neon)
