import numpy as np
import pytest
from srfforge import BandConvolver, NEON, EMIT


def make_neon(n=426):
    return NEON(wavelengths=np.linspace(381, 2511, n), fwhm=np.full(n, 5.0))


class TestBandConvolver:
    def test_output_shape_1d(self):
        neon = make_neon()
        emit = EMIT()
        conv = BandConvolver(source=neon, target=emit)
        refl = np.random.rand(len(neon.wavelengths))
        out = conv(refl)
        assert out.shape == (len(emit.wavelengths),)

    def test_output_shape_3d(self):
        neon = make_neon()
        emit = EMIT()
        conv = BandConvolver(source=neon, target=emit)
        refl = np.random.rand(10, 10, len(neon.wavelengths))
        out = conv(refl)
        assert out.shape == (10, 10, len(emit.wavelengths))

    def test_flat_spectrum_preserved(self):
        neon = make_neon()
        emit = EMIT()
        # Only test bands where EMIT overlaps with NEON
        conv = BandConvolver(source=neon, target=emit)
        flat = np.ones(len(neon.wavelengths))
        out = conv(flat)
        # Bands in the overlapping region should be ~1.0
        neon_min, neon_max = neon.wavelengths[0], neon.wavelengths[-1]
        mask = (emit.wavelengths >= neon_min + 30) & (emit.wavelengths <= neon_max - 30)
        np.testing.assert_allclose(out[mask], 1.0, atol=1e-4)

    def test_matrix_property(self):
        neon = make_neon(100)
        emit = EMIT()
        conv = BandConvolver(source=neon, target=emit)
        assert conv.matrix.shape == (len(emit.wavelengths), 100)

    def test_repr(self):
        conv = BandConvolver(source=make_neon(), target=EMIT())
        assert "BandConvolver" in repr(conv)
        assert "NEON" in repr(conv)
        assert "EMIT" in repr(conv)

    def test_values_in_range(self):
        neon = make_neon()
        emit = EMIT()
        conv = BandConvolver(source=neon, target=emit)
        refl = np.clip(np.random.rand(len(neon.wavelengths)), 0, 1)
        out = conv(refl)
        assert np.all(out >= 0)
        assert np.all(out <= 1.0 + 1e-6)
