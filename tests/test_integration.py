"""End-to-end pipeline tests and cross-validation against spectral.BandResampler."""
import numpy as np
import pytest
from srfforge import BandConvolver, NEON, EMIT
from srfforge.io import read_neon_h5, read_neon_envi


class TestHDF5Pipeline:
    def test_read_and_convolve(self, neon_h5_file):
        refl, neon, meta = read_neon_h5(neon_h5_file)
        assert refl.shape == (10, 10, 426)
        assert refl.dtype == np.float32

        conv = BandConvolver(source=neon, target=EMIT())
        out = conv(refl)
        assert out.shape == (10, 10, len(EMIT().wavelengths))

    def test_metadata_site(self, neon_h5_file):
        _, _, meta = read_neon_h5(neon_h5_file)
        assert meta["site"] == "HARV"
        assert meta["scale_factor"] == 10000.0

    def test_reflectance_range(self, neon_h5_file):
        refl, neon, _ = read_neon_h5(neon_h5_file)
        conv = BandConvolver(source=neon, target=EMIT())
        out = conv(refl)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestENVIPipeline:
    def test_read_shape(self, neon_envi_file, neon_wavelengths):
        refl, neon, meta = read_neon_envi(neon_envi_file)
        assert refl.shape[2] == len(neon_wavelengths)
        assert refl.dtype == np.float32

    def test_wavelengths_match(self, neon_envi_file, neon_wavelengths):
        _, neon, _ = read_neon_envi(neon_envi_file)
        np.testing.assert_allclose(neon.wavelengths, neon_wavelengths, atol=0.01)

    def test_read_and_convolve(self, neon_envi_file):
        refl, neon, _ = read_neon_envi(neon_envi_file)
        conv = BandConvolver(source=neon, target=EMIT())
        out = conv(refl)
        assert out.shape[2] == len(EMIT().wavelengths)

    def test_reflectance_range(self, neon_envi_file):
        refl, neon, _ = read_neon_envi(neon_envi_file)
        conv = BandConvolver(source=neon, target=EMIT())
        out = conv(refl)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestValidationVsSpectral:
    """Cross-validate our convolution against spectral.BandResampler."""

    @pytest.fixture
    def resampler(self, neon_wavelengths, neon_fwhm):
        from spectral.algorithms.resampling import BandResampler  # type: ignore

        emit = EMIT()
        return BandResampler(
            neon_wavelengths,
            emit.wavelengths,
            neon_fwhm,
            emit.fwhm,
        )

    def test_flat_spectrum(self, neon_wavelengths, neon_fwhm, resampler):
        emit = EMIT()
        neon = NEON(wavelengths=neon_wavelengths, fwhm=neon_fwhm)
        conv = BandConvolver(source=neon, target=emit)

        flat = np.ones(len(neon_wavelengths), dtype=np.float32)
        ours = conv(flat)
        theirs = resampler(flat)

        # Check in the well-overlapping interior (avoid edge bands)
        mask = (emit.wavelengths > 450) & (emit.wavelengths < 2400)
        np.testing.assert_allclose(ours[mask], theirs[mask], rtol=0.02, atol=1e-3)

    def test_smooth_spectrum(self, neon_wavelengths, neon_fwhm, synthetic_spectrum, resampler):
        emit = EMIT()
        neon = NEON(wavelengths=neon_wavelengths, fwhm=neon_fwhm)
        conv = BandConvolver(source=neon, target=emit)

        ours = conv(synthetic_spectrum)
        theirs = resampler(synthetic_spectrum.astype(float))

        mask = (emit.wavelengths > 450) & (emit.wavelengths < 2400)
        np.testing.assert_allclose(ours[mask], theirs[mask], rtol=0.02, atol=1e-3)

    def test_correlation(self, neon_wavelengths, neon_fwhm, synthetic_spectrum, resampler):
        """Outputs should be highly correlated even at band edges."""
        emit = EMIT()
        neon = NEON(wavelengths=neon_wavelengths, fwhm=neon_fwhm)
        conv = BandConvolver(source=neon, target=emit)

        ours = conv(synthetic_spectrum)
        theirs = np.array(resampler(synthetic_spectrum.astype(float)))
        # spectral returns NaN for bands with no source overlap; exclude those
        valid = ~np.isnan(theirs)
        corr = np.corrcoef(ours[valid], theirs[valid])[0, 1]
        assert corr > 0.9999
