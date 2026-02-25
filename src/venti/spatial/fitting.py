"""Windowed polynomial plane fitting for InSAR calibration.

Implements a moving-window least-squares surface fit to remove the
long-wavelength bias between InSAR displacement and a GNSS reference surface.
Ported and cleaned up from the ``calibrate_velocity`` script.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg
import scipy.sparse as spu
from joblib import Parallel, delayed
from skimage.filters import gaussian

logger = logging.getLogger(__name__)

# Polynomial order → number of coefficients
_POLY_N_COEFF = {0: 1, 1: 3, 1.5: 4, 2: 6, 3: 8}


# Gap filling
def _fill_gaps(
    array: np.ndarray,
    fill_value: float = 0,
    smoothing_iterations: int = 0,
) -> np.ndarray:
    """Fill zero/NaN gaps via nearest-neighbour propagation.

    Uses GDAL's ``FillNodata`` when available (preferred for large arrays
    with smoothing), otherwise falls back to
    :func:`scipy.ndimage.distance_transform_edt`.

    Parameters
    ----------
    array : np.ndarray
        2-D input array. Gaps are pixels equal to `fill_value` or NaN.
    fill_value : float, optional
        Sentinel value treated as missing, by default ``0``.
    smoothing_iterations : int, optional
        Number of smoothing passes applied after gap filling, by default ``0``.

    Returns
    -------
    np.ndarray
        Gap-filled array.

    """
    try:
        from osgeo import gdal  # noqa: F401

        return _fill_gaps_gdal(array, fill_value, smoothing_iterations)
    except ImportError:
        pass
    return _fill_gaps_scipy(array, fill_value, smoothing_iterations)


def _fill_gaps_gdal(
    array: np.ndarray,
    fill_value: float = 0,
    smoothing_iterations: int = 0,
) -> np.ndarray:
    from osgeo import gdal

    driver = gdal.GetDriverByName("MEM")
    rows, cols = array.shape
    dataset = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    band.WriteArray(array.astype(np.float32))
    band.SetNoDataValue(fill_value)
    gdal.FillNodata(
        targetBand=band,
        maskBand=None,
        maxSearchDist=int(np.max(array.shape) // 4),
        smoothingIterations=smoothing_iterations,
    )
    result = band.ReadAsArray()
    dataset = None  # flush
    return result


def _fill_gaps_scipy(
    array: np.ndarray,
    fill_value: float = 0,
    smoothing_iterations: int = 0,
) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    mask = (array == fill_value) | np.isnan(array)
    if not mask.any():
        return array
    _, nearest_idx = distance_transform_edt(mask, return_indices=True)
    result = array[tuple(nearest_idx)]
    if smoothing_iterations > 0:
        result = gaussian(result, sigma=smoothing_iterations)
    return result


# Window helpers
def _find_data_extent(array: np.ndarray, axis: int = 0) -> tuple[int, int]:
    """Return the start and stop indices of non-zero data along an axis.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    axis : int, optional
        Axis to collapse before counting non-zeros, by default ``0``.

    Returns
    -------
    tuple of int
        ``(start, stop)`` indices.

    """
    count = np.count_nonzero(array, axis=axis)
    nonzero = np.where(count != 0)[0]
    return int(nonzero.min()), int(nonzero.max())


def _get_sliding_windows(
    length: int,
    win_size: int,
    win_overlap: int,
    first: int = 0,
    end: int = 0,
) -> list[slice]:
    """Build a list of overlapping 1-D window slices.

    Parameters
    ----------
    length : int
        Total length of the dimension.
    win_size : int
        Window size in pixels.
    win_overlap : int
        Overlap between adjacent windows in pixels.
    first : int, optional
        Start index, by default ``0``.
    end : int, optional
        End index; ``0`` means use `length`, by default ``0``.

    Returns
    -------
    list of slice
        Window slices covering ``[first, end)``.

    """
    if end == 0:
        end = length

    windows = []
    ix = 1
    stop = win_size
    while stop < end:
        start = first + (ix - 1) * win_size - win_overlap * (ix - 1)
        stop = first + ix * win_size - win_overlap * (ix - 1)
        ix += 1
        windows.append(slice(start, stop))

    if windows:
        windows[-1] = slice(windows[-1].start, end)
    return windows


def _extend_window(
    slice_y: slice,
    slice_x: slice,
    extend_y: int,
    extend_x: int,
    length: int,
    width: int,
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Expand a window by padding and return the extended slice and inner slice.

    Parameters
    ----------
    slice_y : slice
        Row slice of the original window.
    slice_x : slice
        Column slice of the original window.
    extend_y : int
        Pixels to extend in the row direction.
    extend_x : int
        Pixels to extend in the column direction.
    length : int
        Total number of rows.
    width : int
        Total number of columns.

    Returns
    -------
    extended_win : tuple of slice
        ``(row_slice, col_slice)`` for the extended window.
    padding : tuple of slice
        ``(row_slice, col_slice)`` indexing the original window inside the
        extended window.

    """
    y_start = max(slice_y.start - extend_y, 0)
    y_stop = min(slice_y.stop + extend_y, length)
    x_start = max(slice_x.start - extend_x, 0)
    x_stop = min(slice_x.stop + extend_x, width)

    extended_win = (slice(y_start, y_stop), slice(x_start, x_stop))
    nlength = y_stop - y_start
    nwidth = x_stop - x_start
    padding = (
        slice(slice_y.start - y_start, nlength - (y_stop - slice_y.stop)),
        slice(slice_x.start - x_start, nwidth - (x_stop - slice_x.stop)),
    )
    return extended_win, padding


# Plane fitting
def _design_matrix_poly(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1,
    poly_order: float = 1.0,
) -> np.ndarray:
    """Build the polynomial design matrix for plane fitting.

    Parameters
    ----------
    x : np.ndarray
        1-D array of x (easting/longitude) coordinates.
    y : np.ndarray
        1-D array of y (northing/latitude) coordinates.
    c : float, optional
        Constant offset scale, by default ``1``.
    poly_order : float, optional
        Polynomial order: ``0``, ``1``, ``1.5``, ``2``, or ``3``.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(n_obs, n_coeff)``.

    """
    ones = np.ones(len(x)) * c
    if poly_order == 0:
        A = np.array([ones])
    elif poly_order == 1:
        A = np.array([x, y, ones])
    elif poly_order == 1.5:
        A = np.array([x, y, x * y, ones])
    elif poly_order == 2:
        A = np.array([x**2, y**2, x * y, x, y, ones])
    elif poly_order == 3:
        A = np.array([x**3, y**3, x**2, y**2, x * y, x, y, ones])
    else:
        msg = f"Unsupported poly_order={poly_order}. Choose from {list(_POLY_N_COEFF)}"
        raise ValueError(msg)
    return A.T


def _calc_plane_values(
    x: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    poly_order: float = 1.0,
) -> np.ndarray:
    """Evaluate a fitted polynomial surface at given coordinates.

    Parameters
    ----------
    x : np.ndarray
        X coordinates (same shape as output).
    y : np.ndarray
        Y coordinates.
    coef : np.ndarray
        Polynomial coefficients from :func:`_weighted_lscov`.
    poly_order : float, optional
        Polynomial order, by default ``1.0``.

    Returns
    -------
    np.ndarray
        Evaluated surface values.

    """
    if poly_order == 0:
        return np.ones(x.shape) * coef[0]
    if poly_order == 1:
        return x * coef[0] + y * coef[1] + coef[2]
    if poly_order == 1.5:
        return x * coef[0] + y * coef[1] + x * y * coef[2] + coef[3]
    if poly_order == 2:
        return (
            x**2 * coef[0] + y**2 * coef[1]
            + x * y * coef[2] + x * coef[3]
            + y * coef[4] + coef[5]
        )
    if poly_order == 3:
        return (
            x**3 * coef[0] + y**3 * coef[1]
            + x**2 * coef[2] + y**2 * coef[3]
            + x * y * coef[4] + x * coef[5]
            + y * coef[6]
        )
    msg = f"Unsupported poly_order={poly_order}"
    raise ValueError(msg)


def _calc_plane_uncertainty(
    x: np.ndarray,
    y: np.ndarray,
    Qxx: np.ndarray,
    poly_order: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate coefficient covariance to surface uncertainty.

    Parameters
    ----------
    x : np.ndarray
        X coordinates.
    y : np.ndarray
        Y coordinates.
    Qxx : np.ndarray
        Coefficient covariance matrix from :func:`_weighted_lscov`.
    poly_order : float, optional
        Polynomial order, by default ``1.0``.

    Returns
    -------
    plane_var : np.ndarray
        Propagated variance.
    plane_std : np.ndarray
        Propagated standard deviation.

    """
    e = np.ones(x.shape)
    if poly_order == 0:
        plane_var = np.ones(x.shape) * Qxx[0, 0]
    elif poly_order == 1:
        x_var, y_var, z_var = np.diag(Qxx)
        cxy, cxz, cyz = Qxx[0, 1], Qxx[0, 2], Qxx[1, 2]
        plane_var = (
            x**2 * x_var + y**2 * y_var + e**2 * z_var
            + 2 * x * (y * cxy + e * cxz)
            + 2 * y * (e * cyz)
        )
    elif poly_order == 1.5:
        x_var, y_var, xy_var, z_var = np.diag(Qxx)
        cxy, cxxy, cxz = Qxx[0, 1], Qxx[0, 2], Qxx[0, 3]
        cyxy, cyz, cxyz = Qxx[1, 2], Qxx[1, 3], Qxx[2, 3]
        plane_var = (
            x**2 * x_var + y**2 * y_var + (x * y)**2 * xy_var + e**2 * z_var
            + 2 * x * (y * cxy + x * y * cxxy + e * cxz)
            + 2 * y * (x * y * cyxy + e * cyz)
            + 2 * x * y * (e * cxyz)
        )
    else:
        # Higher orders: diagonal-only approximation
        diag = np.diag(Qxx)
        A = _design_matrix_poly(x.ravel(), y.ravel(), poly_order=poly_order)
        plane_var = (A**2 @ diag).reshape(x.shape)

    return plane_var, np.sqrt(np.abs(plane_var))


def _weighted_lscov(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least-squares via QR factorisation (port of MATLAB ``lscov``).

    Parameters
    ----------
    A : np.ndarray
        Design matrix of shape ``(n_obs, n_params)``.
    b : np.ndarray
        Observation vector of length ``n_obs``.
    weights : np.ndarray, optional
        Per-observation weights (inverse-variance). Uniform by default.

    Returns
    -------
    x : np.ndarray
        Estimated parameters, shape ``(n_params, 1)``.
    stdx : np.ndarray
        Standard errors of the parameters.
    mse : float
        Mean squared error.
    Qxx : np.ndarray
        Covariance matrix of the parameters.
    res : np.ndarray
        Unweighted residuals.
    wres : np.ndarray
        Weighted residuals.
    obs_hat : np.ndarray
        Fitted observations.
    Qcov : np.ndarray
        Aposteriori cofactor matrix.

    """
    if weights is None:
        weights = np.ones(b.shape)

    b = b[:, np.newaxis]
    n_obs, n_x = A.shape
    n_r = n_obs - n_x

    if spu.issparse(A):
        A = A.toarray()

    Aw = A * np.sqrt(weights[:, np.newaxis])
    Bw = b * np.sqrt(weights[:, np.newaxis])

    Q, R, perm = scipy.linalg.qr(Aw, mode="economic", pivoting=True)
    z = Q.T @ Bw

    r_diag = np.diag(R)
    keep = np.abs(r_diag) > np.abs(r_diag[0]) * max(n_obs, n_x) * np.finfo(R.dtype).eps
    rank = int(keep.sum())
    if rank < n_x:
        logger.warning("Design matrix rank-deficient: %d / %d columns kept", rank, n_x)
        R = R[np.ix_(keep, keep)]
        z = z[keep, :]
        perm = perm[keep]

    xx = np.linalg.lstsq(R, z, rcond=None)[0]
    x = np.zeros((n_x, 1))
    x[perm] = xx

    Q_mat = Q if rank == n_x else Q[:, keep]
    wres = Bw - Q_mat @ z
    mse = float(np.sum(wres * wres.conj()) / n_r) if n_r > 0 else 0.0

    Rinv = np.triu(np.linalg.lstsq(R, np.eye(rank), rcond=None)[0])
    Qxx = np.zeros((n_x, n_x))
    Qxx[np.ix_(perm, perm)] = Rinv @ Rinv.T

    stdx = np.sqrt(mse * np.diag(Qxx))
    res = wres / np.sqrt(weights[:, np.newaxis])
    obs_hat = Q_mat @ z / np.sqrt(weights[:, np.newaxis])
    Qcov = A @ (Qxx / mse) @ A.T if n_r > 0 else np.zeros(Qxx.shape)

    return x, stdx, mse, Qxx, res, wres, obs_hat, Qcov


def _fit_plane(
    data: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    order: float = 1.5,
    decimate: int = 1,
    smooth: bool = False,
    smooth_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a polynomial surface to 2-D data at given coordinates.

    Parameters
    ----------
    data : np.ndarray
        2-D data array.
    lons : np.ndarray
        2-D array of x/longitude coordinates (same shape as `data`).
    lats : np.ndarray
        2-D array of y/latitude coordinates (same shape as `data`).
    order : float, optional
        Polynomial order, by default ``1.5``.
    decimate : int, optional
        Subsampling factor for the inversion, by default ``1``.
    smooth : bool, optional
        Apply Gaussian smoothing before fitting, by default ``False``.
    smooth_sigma : float, optional
        Sigma for Gaussian smoothing in pixels, by default ``5.0``.

    Returns
    -------
    plane : np.ndarray
        Fitted surface, same shape as `data`.
    plane_std : np.ndarray
        Propagated uncertainty of the fit.

    """
    nan_idx = np.isnan(data.ravel())

    if smooth:
        data = gaussian(data, smooth_sigma)

    A = _design_matrix_poly(lons.ravel(), lats.ravel(), poly_order=order)
    A_clean = np.delete(A, nan_idx, axis=0)
    b_clean = np.delete(data.ravel(), nan_idx)
    w = np.ones_like(b_clean)

    _, _, _, Qxx, res, *_ = _weighted_lscov(
        A_clean[::decimate], b_clean[::decimate], w[::decimate]
    )

    # Remove 2-sigma outliers and refit
    res = res.ravel()
    outliers = (res < res.mean() - 2 * res.std()) | (res > res.mean() + 2 * res.std())
    A_ref = np.delete(A_clean, np.where(outliers), axis=0)
    b_ref = np.delete(b_clean, np.where(outliers))
    w_ref = np.ones_like(b_ref)

    x1, _, _, Qxx, *_ = _weighted_lscov(
        A_ref[::decimate], b_ref[::decimate], w_ref[::decimate]
    )

    plane = _calc_plane_values(lons, lats, x1.ravel(), order)
    _, plane_std = _calc_plane_uncertainty(lons, lats, Qxx, order)
    return plane, plane_std


def _get_residual_mask(
    insar_data: np.ndarray,
    gnss_los: np.ndarray,
    lower_quantile: float = 0.15,
    upper_quantile: float = 0.85,
) -> np.ndarray:
    """Mask pixels whose InSAR–GNSS residual is an outlier.

    Parameters
    ----------
    insar_data : np.ndarray
        InSAR displacement or velocity field.
    gnss_los : np.ndarray
        GNSS LOS reference field.
    lower_quantile : float, optional
        Lower residual quantile threshold, by default ``0.15``.
    upper_quantile : float, optional
        Upper residual quantile threshold, by default ``0.85``.

    Returns
    -------
    np.ndarray
        Boolean mask: ``True`` where pixels should be excluded.

    """
    invalid = np.ma.masked_invalid(insar_data).mask
    residual_filled = np.where(invalid, np.nan, insar_data - gnss_los)
    lo = np.nanquantile(residual_filled, lower_quantile)
    hi = np.nanquantile(residual_filled, upper_quantile)
    return invalid | (residual_filled < lo) | (residual_filled > hi)


def _get_coordinate_grid(
    snwe: tuple[float, float, float, float],
    win_y: list[int],
    win_x: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a coordinate meshgrid for a window region.

    Parameters
    ----------
    snwe : tuple of float
        ``(south, north, west, east)`` bounds.
    win_y : list of int
        ``[y_start, y_stop]`` pixel row indices.
    win_x : list of int
        ``[x_start, x_stop]`` pixel column indices.

    Returns
    -------
    grid_lons : np.ndarray
    grid_lats : np.ndarray

    """
    lons = np.linspace(snwe[2], snwe[3], win_x[1] - win_x[0])
    lats = np.linspace(snwe[1], snwe[0], win_y[1] - win_y[0])
    return np.meshgrid(lons, lats)


def _process_window(
    win_index: tuple[slice, slice],
    insar_data: np.ndarray,
    gnss_los: np.ndarray,
    snwe: tuple[float, float, float, float],
    win_extend_y: int,
    win_extend_x: int,
    length: int,
    width: int,
    poly_order: float,
) -> tuple[tuple[slice, slice], np.ndarray | None, np.ndarray | None]:
    """Fit a calibration plane within a single moving window.

    Designed to be called in parallel via ``joblib``.

    Parameters
    ----------
    win_index : tuple of slice
        ``(row_slice, col_slice)`` of the window to fill.
    insar_data : np.ndarray
        Full InSAR data array (gap-filled, masked).
    gnss_los : np.ndarray
        Full GNSS LOS reference array.
    snwe : tuple of float
        Geographic bounds ``(south, north, west, east)``.
    win_extend_y : int
        Row extension for fitting context.
    win_extend_x : int
        Column extension for fitting context.
    length : int
        Total number of rows.
    width : int
        Total number of columns.
    poly_order : float
        Polynomial order for plane fitting.

    Returns
    -------
    win_index : tuple of slice
    plane : np.ndarray or None
    plane_std : np.ndarray or None

    """
    win2, pad = _extend_window(
        win_index[0], win_index[1], win_extend_y, win_extend_x, length, width
    )
    win_y = [win2[0].start, win2[0].stop]
    win_x = [win2[1].start, win2[1].stop]
    win_lons, win_lats = _get_coordinate_grid(snwe, win_y=win_y, win_x=win_x)

    res = insar_data[win2] - gnss_los[win2]
    if np.isnan(res).sum() / res.size > 0.8:
        return win_index, None, None

    try:
        plane, plane_std = _fit_plane(res, win_lons, win_lats, order=poly_order, decimate=50)
        mask2 = np.ma.masked_invalid(res).mask
        filled_plane = np.ma.masked_array(plane[pad], mask=mask2[pad]).filled(0)
        filled_std = np.ma.masked_array(plane_std[pad], mask=mask2[pad]).filled(0)
        return win_index, filled_plane, filled_std
    except Exception:
        logger.debug("Skipping window %s", win_index, exc_info=True)
        return win_index, None, None


def fit_windowed_plane(
    insar_data: np.ndarray,
    gnss_los: np.ndarray,
    win_xsize: int,
    win_ysize: int,
    win_overlap_x: int,
    win_overlap_y: int,
    win_extend_x: int,
    win_extend_y: int,
    snwe: tuple[float, float, float, float],
    gnss_los_std: np.ndarray | None = None,
    poly_order: float = 1.5,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a windowed polynomial calibration surface to InSAR data.

    Uses a moving-window least-squares approach to estimate the
    long-wavelength surface ``m`` such that ``insar_data ≈ gnss_los + m``.
    The correction is applied as ``calibrated = insar_data - m``.

    Parameters
    ----------
    insar_data : np.ndarray
        2-D InSAR displacement or velocity array.
    gnss_los : np.ndarray
        2-D GNSS LOS reference field (same shape as `insar_data`).
    win_xsize : int
        Window width in pixels.
    win_ysize : int
        Window height in pixels.
    win_overlap_x : int
        Window overlap in the x direction.
    win_overlap_y : int
        Window overlap in the y direction.
    win_extend_x : int
        Extension beyond the window for fitting context.
    win_extend_y : int
        Extension beyond the window for fitting context.
    snwe : tuple of float
        Geographic bounds ``(south, north, west, east)`` used to build the
        coordinate grid for polynomial fitting.
    gnss_los_std : np.ndarray, optional
        Uncertainty of the GNSS LOS field (reserved for future weighted
        inversion; currently unused).
    poly_order : float, optional
        Polynomial order for plane fitting, by default ``1.5``.
    n_jobs : int, optional
        Number of parallel jobs for ``joblib.Parallel``,
        by default ``-1`` (all CPUs).

    Returns
    -------
    calibration_surface : np.ndarray
        Estimated calibration surface to subtract from `insar_data`.
    calibration_std : np.ndarray
        Uncertainty of the calibration surface.

    Examples
    --------
    ::

        surface, surface_std = fit_windowed_plane(
            insar_data=displacement_mm,
            gnss_los=gnss_los_mm,
            win_xsize=1000, win_ysize=1000,
            win_overlap_x=10, win_overlap_y=10,
            win_extend_x=1000, win_extend_y=1000,
            snwe=(3800000, 3900000, 400000, 500000),
        )
        calibrated = displacement_mm - surface

    """
    outlier_mask = _get_residual_mask(insar_data, gnss_los)
    invalid_mask = np.ma.masked_invalid(insar_data).mask
    length, width = insar_data.shape

    logger.info(
        "Fitting windowed calibration plane: %d x %d px windows, poly_order=%s",
        win_ysize, win_xsize, poly_order,
    )

    # Gap-fill before fitting
    insar_filled = np.ma.masked_array(insar_data, mask=outlier_mask).filled(0)
    insar_filled = _fill_gaps(insar_filled, fill_value=0, smoothing_iterations=10)
    insar_filled = np.ma.masked_array(insar_filled, invalid_mask)

    # Build window index list
    y_start, y_stop = _find_data_extent(insar_data, axis=1)
    win_ys = _get_sliding_windows(length, win_ysize, win_overlap_y, y_start, y_stop)

    all_windows: list[tuple[slice, slice]] = []
    for win_y in win_ys:
        x_start, x_stop = _find_data_extent(insar_data[win_y, :], axis=0)
        win_xs = _get_sliding_windows(
            x_stop - x_start, win_xsize, win_overlap_x, x_start, x_stop
        )
        all_windows.extend((win_y, win_x) for win_x in win_xs)

    logger.info("Processing %d windows (n_jobs=%d)", len(all_windows), n_jobs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_window)(
            ix, insar_filled, gnss_los, snwe,
            win_extend_y, win_extend_x, length, width, poly_order,
        )
        for ix in all_windows
    )

    cal_surface = np.zeros(insar_data.shape, dtype=np.float32)
    cal_std = np.zeros(insar_data.shape, dtype=np.float32)
    for ix, plane_val, std_val in results:
        if plane_val is not None:
            cal_surface[ix] = plane_val
            cal_std[ix] = std_val

    return cal_surface, cal_std
