import numpy as np
import pytest
from srfforge.srf import build_convolution_matrix


def test_matrix_shape():
    src_wl = np.linspace(400, 2500, 426)
    tgt_wl = np.linspace(400, 2500, 50)
    tgt_fwhm = np.full(50, 10.0)
    M = build_convolution_matrix(src_wl, tgt_wl, tgt_fwhm)
    assert M.shape == (50, 426)


def test_rows_sum_to_one():
    src_wl = np.linspace(400, 2500, 426)
    tgt_wl = np.linspace(500, 2400, 30)
    tgt_fwhm = np.full(30, 15.0)
    M = build_convolution_matrix(src_wl, tgt_wl, tgt_fwhm)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-6)


def test_identity_case():
    """When source and target share the same bands, M should be close to identity."""
    wl = np.linspace(500, 2000, 100)
    # Very narrow FWHM relative to band spacing
    fwhm = np.full(100, 0.01)
    M = build_convolution_matrix(wl, wl, fwhm)
    np.testing.assert_allclose(M, np.eye(100), atol=1e-4)


def test_convolution_preserves_flat_spectrum():
    """A flat (constant) spectrum should remain flat after convolution."""
    src_wl = np.linspace(400, 2500, 426)
    tgt_wl = np.linspace(500, 2400, 100)
    tgt_fwhm = np.full(100, 10.0)
    M = build_convolution_matrix(src_wl, tgt_wl, tgt_fwhm)

    flat = np.ones(426)
    result = M @ flat
    np.testing.assert_allclose(result, np.ones(100), atol=1e-6)
