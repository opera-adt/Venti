"""LOS projection for GNSS displacement data.

Projects GNSS east/north/up observations into the InSAR line-of-sight
direction and interpolates the result onto the full raster grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ..spatial.interpolation import (
    InterpolationMethod,
    RbfFunction,
    _regular_grid_interpolator_from_ds,
    _sample_on_points,
    interpolate_griddata,
    interpolate_rbf,
)

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)


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
        RBF kernel passed to :func:`~venti.spatial.interpolation.interpolate_rbf`
        when ``method='rbf'``, by default ``'cubic'``.

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
