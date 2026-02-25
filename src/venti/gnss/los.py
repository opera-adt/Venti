"""LOS projection and spatial interpolation for GNSS displacement data.

Projects GNSS east/north/up observations into the InSAR line-of-sight
direction and interpolates the result onto the full raster grid.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from scipy.interpolate import Rbf, RegularGridInterpolator, griddata
from tqdm import tqdm

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)

InterpolationMethod = Literal["rbf", "griddata"]
RbfFunction = Literal["multiquadric", "inverse", "gaussian", "linear", "cubic", "quintic", "thin_plate"]
GriddataMethod = Literal["linear", "nearest", "cubic"]


def _regular_grid_interpolator_from_ds(
    ds: xr.Dataset,
    arr: np.ndarray,
) -> RegularGridInterpolator:
    """Build a (y, x) regular-grid interpolator from an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``'x'`` and ``'y'`` coordinate arrays.
    arr : np.ndarray
        2-D array of shape ``(len(y), len(x))``.

    Returns
    -------
    RegularGridInterpolator
        Interpolator that accepts ``[[y, x], ...]`` query points.

    """
    x = np.asarray(ds["x"].values)
    y = np.asarray(ds["y"].values)

    assert arr.shape == (y.size, x.size), (  # noqa: S101
        f"Array shape {arr.shape} must match (len(y)={y.size}, len(x)={x.size})"
    )

    # RegularGridInterpolator requires strictly increasing axes
    if np.any(np.diff(y) < 0):
        y = y[::-1]
        arr = arr[::-1, :]
    if np.any(np.diff(x) < 0):
        x = x[::-1]
        arr = arr[:, ::-1]

    return RegularGridInterpolator((y, x), arr, bounds_error=False, fill_value=np.nan)


def _sample_on_points(
    ds: xr.Dataset,
    arr: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
) -> np.ndarray:
    """Sample a raster grid at arbitrary (x, y) point locations.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset defining the ``'x'`` and ``'y'`` coordinate axes.
    arr : np.ndarray
        2-D raster array of shape ``(len(y), len(x))``.
    pts_x : np.ndarray
        X coordinates of query points.
    pts_y : np.ndarray
        Y coordinates of query points.

    Returns
    -------
    np.ndarray
        Sampled values at each query point.

    """
    interp = _regular_grid_interpolator_from_ds(ds, arr)
    return interp(np.column_stack([pts_y, pts_x]))


def interpolate_rbf(
    netcdf_file: str | Path,
    gx: np.ndarray,
    gy: np.ndarray,
    zv: np.ndarray,
    function: RbfFunction = "cubic",
    tile_nx: int = 512,
    tile_ny: int = 512,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Interpolate scattered GNSS samples onto a regular grid using RBF.

    Uses :class:`scipy.interpolate.Rbf` with tiled evaluation to limit
    peak memory usage on large grids.

    Parameters
    ----------
    netcdf_file : str or Path
        NetCDF file whose ``'x'`` and ``'y'`` coordinates define the output grid.
    gx : np.ndarray
        X coordinates of the scattered input samples.
    gy : np.ndarray
        Y coordinates of the scattered input samples.
    zv : np.ndarray
        Values at the scattered input samples.
    function : str, optional
        RBF basis function, by default ``'cubic'``.
    tile_nx : int, optional
        Tile width in pixels, by default ``512``.
    tile_ny : int, optional
        Tile height in pixels, by default ``512``.
    dtype : numpy dtype, optional
        Output dtype, by default ``np.float32``.

    Returns
    -------
    np.ndarray
        Interpolated grid of shape ``(ny, nx)``.

    """
    with xr.open_dataset(netcdf_file) as ds:
        x = np.asarray(ds["x"].values, dtype=dtype)
        y = np.asarray(ds["y"].values, dtype=dtype)

    nx, ny = x.size, y.size
    Z = np.full((ny, nx), np.nan, dtype=dtype)
    rbf = Rbf(gx, gy, zv, function=function, smooth=1)

    nx_tiles = math.ceil(nx / tile_nx)
    ny_tiles = math.ceil(ny / tile_ny)

    with tqdm(total=nx_tiles * ny_tiles, desc="RBF interpolation") as pbar:
        for j in range(ny_tiles):
            y0, y1 = j * tile_ny, min((j + 1) * tile_ny, ny)
            y_slice = y[y0:y1]

            for i in range(nx_tiles):
                x0, x1 = i * tile_nx, min((i + 1) * tile_nx, nx)
                x_slice = x[x0:x1]

                xv, yv = np.meshgrid(x_slice, y_slice, indexing="xy")
                Z[y0:y1, x0:x1] = rbf(xv, yv).reshape(y_slice.size, x_slice.size)
                pbar.update(1)

    return Z


def interpolate_griddata(
    netcdf_file: str | Path,
    gx: np.ndarray,
    gy: np.ndarray,
    zv: np.ndarray,
    method: GriddataMethod = "linear",
    tile_nx: int = 512,
    tile_ny: int = 512,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Interpolate scattered GNSS samples onto a regular grid using griddata.

    Parameters
    ----------
    netcdf_file : str or Path
        NetCDF file whose ``'x'`` and ``'y'`` coordinates define the output grid.
    gx : np.ndarray
        X coordinates of the scattered input samples.
    gy : np.ndarray
        Y coordinates of the scattered input samples.
    zv : np.ndarray
        Values at the scattered input samples.
    method : str, optional
        Interpolation method passed to :func:`scipy.interpolate.griddata`,
        by default ``'linear'``.
    tile_nx : int, optional
        Tile width in pixels, by default ``512``.
    tile_ny : int, optional
        Tile height in pixels, by default ``512``.
    dtype : numpy dtype, optional
        Output dtype, by default ``np.float32``.

    Returns
    -------
    np.ndarray
        Interpolated grid of shape ``(ny, nx)``.

    """
    with xr.open_dataset(netcdf_file) as ds:
        x = np.asarray(ds["x"].values, dtype=dtype)
        y = np.asarray(ds["y"].values, dtype=dtype)

    nx, ny = x.size, y.size
    Z = np.full((ny, nx), np.nan, dtype=dtype)
    points = np.column_stack((gx, gy))
    values = np.asarray(zv, dtype=dtype)

    nx_tiles = math.ceil(nx / tile_nx)
    ny_tiles = math.ceil(ny / tile_ny)

    with tqdm(total=nx_tiles * ny_tiles, desc="griddata interpolation") as pbar:
        for j in range(ny_tiles):
            y0, y1 = j * tile_ny, min((j + 1) * tile_ny, ny)
            y_slice = y[y0:y1]

            for i in range(nx_tiles):
                x0, x1 = i * tile_nx, min((i + 1) * tile_nx, nx)
                x_slice = x[x0:x1]

                xv, yv = np.meshgrid(x_slice, y_slice, indexing="xy")
                coords = np.column_stack((xv.ravel(), yv.ravel()))
                z_tile = griddata(points, values, coords, method=method)
                Z[y0:y1, x0:x1] = z_tile.reshape(y_slice.size, x_slice.size)
                pbar.update(1)

    return Z


def project_to_los(
    los_east: np.ndarray,
    los_north: np.ndarray,
    los_up: np.ndarray,
    netcdf_file: str | Path,
    gnss_gdf: gpd.GeoDataFrame,
    method: InterpolationMethod = "rbf",
    rbf_function: RbfFunction = "cubic",
) -> np.ndarray:
    """Project GNSS east/north/up displacements into the InSAR LOS direction.

    Samples the LOS unit vectors at each GNSS station location, forms the
    dot product with the GNSS displacement vector, then interpolates the
    resulting scalar field onto the full raster grid.

    Parameters
    ----------
    los_east : np.ndarray
        2-D array of east LOS unit-vector components, shape ``(ny, nx)``.
    los_north : np.ndarray
        2-D array of north LOS unit-vector components.
    los_up : np.ndarray
        2-D array of up LOS unit-vector components.
    netcdf_file : str or Path
        NetCDF file whose ``'x'`` / ``'y'`` coordinates define the output grid
        and the CRS for the LOS arrays.
    gnss_gdf : gpd.GeoDataFrame
        GNSS station data with a ``geometry`` column (Point, UTM) and columns
        ``'deast'``, ``'dnorth'``, ``'dup'`` containing the displacement or
        velocity in each component.
    method : {'rbf', 'griddata'}, optional
        Spatial interpolation method, by default ``'rbf'``.
    rbf_function : str, optional
        RBF kernel passed to :func:`interpolate_rbf` when ``method='rbf'``,
        by default ``'cubic'``.

    Returns
    -------
    np.ndarray
        LOS scalar field interpolated onto the full raster grid,
        shape ``(ny, nx)``.

    Raises
    ------
    ValueError
        If no valid GNSS LOS samples remain after masking.

    """
    dtype = np.float32

    geo_x = np.asarray([p.x for p in gnss_gdf.geometry], dtype=dtype)
    geo_y = np.asarray([p.y for p in gnss_gdf.geometry], dtype=dtype)
    E = np.asarray(gnss_gdf["deast"].values, dtype=dtype)
    N = np.asarray(gnss_gdf["dnorth"].values, dtype=dtype)
    U = np.asarray(gnss_gdf["dup"].values, dtype=dtype)

    with xr.open_dataset(netcdf_file) as ds:
        de_s = _sample_on_points(ds, np.asarray(los_east, dtype=dtype), geo_x, geo_y)
        dn_s = _sample_on_points(ds, np.asarray(los_north, dtype=dtype), geo_x, geo_y)
        du_s = _sample_on_points(ds, np.asarray(los_up, dtype=dtype), geo_x, geo_y)

    los_at_stations = (de_s * E + dn_s * N + du_s * U).astype(dtype)

    valid = np.isfinite(los_at_stations) & np.isfinite(geo_x) & np.isfinite(geo_y)
    gx = geo_x[valid]
    gy = geo_y[valid]
    zv = los_at_stations[valid]

    if zv.size == 0:
        msg = "No valid GNSS LOS samples after masking — check CRS, coverage, and masks."
        raise ValueError(msg)

    logger.info("Interpolating LOS from %d GNSS stations using method='%s'", zv.size, method)

    if method == "rbf":
        return interpolate_rbf(netcdf_file, gx, gy, zv, function=rbf_function)
    if method == "griddata":
        return interpolate_griddata(netcdf_file, gx, gy, zv)

    msg = f"Unknown interpolation method '{method}'. Choose 'rbf' or 'griddata'."
    raise ValueError(msg)
