import numpy as np


def build_convolution_matrix(
    source_wl: np.ndarray,
    target_wl: np.ndarray,
    target_fwhm: np.ndarray,
) -> np.ndarray:
    """
    Build a normalized Gaussian convolution matrix.

    Parameters
    ----------
    source_wl : (n_source,) wavelengths of the source instrument in nm
    target_wl : (n_target,) band center wavelengths of the target instrument in nm
    target_fwhm : (n_target,) FWHM of each target band in nm

    Returns
    -------
    M : (n_target, n_source) row-normalized convolution matrix
        M[i, j] is the weight of source band j when computing target band i.
    """
    source_wl = np.asarray(source_wl, dtype=float)
    target_wl = np.asarray(target_wl, dtype=float)
    target_fwhm = np.asarray(target_fwhm, dtype=float)

    sigma = target_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # (n_target,)
    diff = source_wl[np.newaxis, :] - target_wl[:, np.newaxis]  # (n_target, n_source)
    M = np.exp(-0.5 * (diff / sigma[:, np.newaxis]) ** 2)

    row_sums = M.sum(axis=1, keepdims=True)
    M /= np.where(row_sums > 0, row_sums, 1.0)
    return M