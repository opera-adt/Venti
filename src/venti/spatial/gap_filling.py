"""Gap filling and outlier masking for raster data."""

from __future__ import annotations

import numpy as np


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

    from ..filtering.gaussian import apply_gaussian

    mask = (array == fill_value) | np.isnan(array)
    if not mask.any():
        return array
    _, nearest_idx = distance_transform_edt(mask, return_indices=True)
    result = array[tuple(nearest_idx)]
    if smoothing_iterations > 0:
        result = apply_gaussian(result, sigma=smoothing_iterations)
    return result


def _get_residual_mask(
    insar_data: np.ndarray,
    gnss_los: np.ndarray,
    lower_quantile: float = 0.15,
    upper_quantile: float = 0.85,
) -> np.ndarray:
    """Mask pixels whose InSAR-GNSS residual is an outlier.

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
