"""GNSSReference: high-level interface for GNSS-based InSAR calibration data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd

from .los import InterpolationMethod, RbfFunction, project_to_los
from .unr import (
    calculate_station_velocity,
    download_grid_lookup,
    download_station,
    find_stations_in_bounds,
    read_epoch_displacements,
)

logger = logging.getLogger(__name__)


@dataclass
class GNSSReference:
    """GNSS grid reference data for a geographic region.

    Downloads UNR GNSS grid timeseries for stations within the specified
    bounds and provides methods for projecting GNSS observations into the
    InSAR line-of-sight (LOS) direction.

    Parameters
    ----------
    bounds : tuple of float
        Bounding box as ``(south, north, west, east)`` in the UTM CRS.
    output_dir : Path
        Directory for downloaded GNSS files.
    reference_frame : str, optional
        GNSS reference frame, ``'IGS20'`` or ``'IGS14'``, by default ``'IGS20'``.
    utm_epsg : int or None, optional
        EPSG code of the UTM projection. Required before calling
        :meth:`download_stations`.

    Attributes
    ----------
    station_files : list of Path
        Paths to downloaded station files (set after :meth:`download_stations`).
    station_gdf : gpd.GeoDataFrame or None
        Filtered station GeoDataFrame in UTM CRS (set after :meth:`download_stations`).

    Examples
    --------
    Download stations and compute a constant LOS velocity field::

        gnss = GNSSReference(
            bounds=(3800000, 3900000, 400000, 500000),
            output_dir=Path('output/GNSS'),
            utm_epsg=32611,
        )
        n = gnss.download_stations()
        gnss_los = gnss.compute_velocity_los(
            los_east, los_north, los_up, 'displacement.nc'
        )

    """

    bounds: tuple[float, float, float, float]
    output_dir: Path
    reference_frame: str = "IGS20"
    utm_epsg: int | None = None

    station_files: list[Path] = field(default_factory=list, init=False)
    station_gdf: gpd.GeoDataFrame | None = field(default=None, init=False)

    def download_stations(self) -> int:
        """Download all GNSS stations within the configured bounds.

        Returns
        -------
        int
            Number of stations downloaded (or already cached).

        Raises
        ------
        ValueError
            If :attr:`utm_epsg` is not set.

        """
        if self.utm_epsg is None:
            msg = "utm_epsg must be set before calling download_stations"
            raise ValueError(msg)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        lookup_path = self.output_dir / "grid_latlon_lookup.txt"
        if not lookup_path.exists():
            lookup_path = download_grid_lookup(self.output_dir, self.reference_frame)

        self.station_gdf = find_stations_in_bounds(
            lookup_path, self.bounds, self.utm_epsg
        )
        logger.info("Found %d GNSS stations in bounds", len(self.station_gdf))

        self.station_files = []
        for station_id in self.station_gdf.index:
            path = download_station(station_id, self.output_dir, self.reference_frame)
            self.station_files.append(path)

        logger.info("Downloaded %d station files", len(self.station_files))
        return len(self.station_files)

    def _build_velocity_gdf(self, start_year: float = 2014.0):
        """Compute velocity GeoDataFrame for all stations.

        Parameters
        ----------
        start_year : float, optional
            Exclude observations before this decimal year, by default ``2014.0``.

        Returns
        -------
        gpd.GeoDataFrame
            Velocity GeoDataFrame with columns ``deast``, ``dnorth``, ``dup``,
            ``dsigma_e``, ``dsigma_n``, ``dsigma_u``, and ``geometry``.

        """
        import geopandas as gpd
        import pandas as pd

        rows = []
        for path in self.station_files:
            station_id = int(path.name.split("_")[0])
            ve, vn, vu, se, sn, su = calculate_station_velocity(path, start_year)
            rows.append(
                {
                    "id": station_id,
                    "deast": ve,
                    "dnorth": vn,
                    "dup": vu,
                    "dsigma_e": se,
                    "dsigma_n": sn,
                    "dsigma_u": su,
                }
            )

        assert self.station_gdf is not None, "Call download_stations() first"
        df = pd.DataFrame(rows).set_index("id")
        return gpd.GeoDataFrame(df.join(self.station_gdf[["geometry"]], how="inner"))

    def compute_velocity_los(
        self,
        los_east: np.ndarray,
        los_north: np.ndarray,
        los_up: np.ndarray,
        netcdf_file: str | Path,
        start_year: float = 2014.0,
        method: InterpolationMethod = "rbf",
        rbf_function: RbfFunction = "cubic",
    ) -> np.ndarray:
        """Compute GNSS velocity projected into the InSAR LOS direction.

        Parameters
        ----------
        los_east : np.ndarray
            2-D east LOS unit-vector component, shape ``(ny, nx)``.
        los_north : np.ndarray
            2-D north LOS unit-vector component.
        los_up : np.ndarray
            2-D up LOS unit-vector component.
        netcdf_file : str or Path
            NetCDF file defining the output raster grid.
        start_year : float, optional
            Earliest observation year used in velocity estimation,
            by default ``2014.0``.
        method : {'rbf', 'griddata'}, optional
            Spatial interpolation method, by default ``'rbf'``.
        rbf_function : str, optional
            RBF basis function when ``method='rbf'``, by default ``'cubic'``.

        Returns
        -------
        np.ndarray
            GNSS LOS velocity field, shape ``(ny, nx)``, in mm/yr.

        """
        assert self.station_files, "Call download_stations() first"
        velocity_gdf = self._build_velocity_gdf(start_year)
        return project_to_los(
            los_east,
            los_north,
            los_up,
            netcdf_file,
            velocity_gdf,
            method=method,
            rbf_function=rbf_function,
        )

    def compute_displacement_los(
        self,
        ref_date: float,
        sec_date: float,
        los_east: np.ndarray,
        los_north: np.ndarray,
        los_up: np.ndarray,
        netcdf_file: str | Path,
        method: InterpolationMethod = "rbf",
        rbf_function: RbfFunction = "cubic",
    ) -> np.ndarray:
        """Compute epoch-specific GNSS displacement projected into LOS.

        Parameters
        ----------
        ref_date : float
            Reference epoch as decimal year.
        sec_date : float
            Secondary epoch as decimal year.
        los_east : np.ndarray
            2-D east LOS unit-vector component.
        los_north : np.ndarray
            2-D north LOS unit-vector component.
        los_up : np.ndarray
            2-D up LOS unit-vector component.
        netcdf_file : str or Path
            NetCDF file defining the output raster grid.
        method : {'rbf', 'griddata'}, optional
            Spatial interpolation method, by default ``'rbf'``.
        rbf_function : str, optional
            RBF basis function when ``method='rbf'``, by default ``'cubic'``.

        Returns
        -------
        np.ndarray
            GNSS LOS displacement field, shape ``(ny, nx)``.

        """
        assert self.station_files, "Call download_stations() first"
        assert self.station_gdf is not None, "Call download_stations() first"

        disp_gdf = read_epoch_displacements(
            self.station_files, ref_date, sec_date, self.station_gdf
        )
        return project_to_los(
            los_east,
            los_north,
            los_up,
            netcdf_file,
            disp_gdf,
            method=method,
            rbf_function=rbf_function,
        )


def compute_gnss_los(
    gnss_ref: GNSSReference,
    los_east: np.ndarray,
    los_north: np.ndarray,
    los_up: np.ndarray,
    netcdf_file: str | Path,
    grid_type: str,
    cache_dir: Path,
    ref_date: float | None = None,
    sec_date: float | None = None,
    starting_year: float = 2014.0,
    recompute: bool = False,
) -> np.ndarray:
    """Compute GNSS LOS displacement or velocity.

    For ``grid_type='constant'`` the function computes a GNSS LOS velocity
    field (mm/yr) and scales it by the epoch interval
    ``(sec_date - ref_date)`` to obtain a displacement in mm.  The velocity
    grid is written once to ``cache_dir/gnss_los_velocity.npy`` and reused
    for every subsequent epoch.

    For ``grid_type='variable'`` an epoch-specific GNSS LOS displacement is
    computed for each ``(ref_date, sec_date)`` pair and cached under
    ``cache_dir/gnss_los_disp_{ref_date:.4f}_{sec_date:.4f}.npy``.

    Parameters
    ----------
    gnss_ref : GNSSReference
        Initialised GNSS reference object with stations already downloaded.
    los_east : np.ndarray
        2-D east LOS unit-vector component, shape ``(ny, nx)``.
    los_north : np.ndarray
        2-D north LOS unit-vector component.
    los_up : np.ndarray
        2-D up LOS unit-vector component.
    netcdf_file : str or Path
        NetCDF file that defines the output raster grid.
    grid_type : str
        ``'constant'`` or ``'variable'``.
    cache_dir : Path
        Directory used for reading and writing ``.npy`` cache files.
    ref_date : float, optional
        Reference epoch as decimal year.  Required for ``'variable'`` and
        for scaling the constant velocity to a displacement.
    sec_date : float, optional
        Secondary epoch as decimal year.  Same requirements as ``ref_date``.
    starting_year : float, optional
        Earliest observation year for velocity estimation
        (``grid_type='constant'`` only), by default ``2014.0``.
    recompute : bool, optional
        When ``True``, ignore any existing cache file and recompute,
        by default ``False``.

    Returns
    -------
    np.ndarray
        GNSS LOS field in mm, shape ``(ny, nx)``.

    """
    logger = logging.getLogger(__name__)

    if grid_type == "constant":
        cache_file = cache_dir / "gnss_los_velocity.npy"
        if cache_file.exists() and not recompute:
            logger.info("Loading cached GNSS LOS velocity from %s", cache_file)
            velocity = np.load(cache_file)
        else:
            logger.info("Computing GNSS LOS velocity from %.4f", starting_year)
            velocity = gnss_ref.compute_velocity_los(
                los_east=los_east,
                los_north=los_north,
                los_up=los_up,
                netcdf_file=netcdf_file,
                start_year=starting_year,
                method="rbf",
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, velocity)
            logger.info("Saved GNSS LOS velocity to %s", cache_file)

        if ref_date is None or sec_date is None:
            return velocity
        # sec_date > ref_date (OPERA convention), so the interval is positive.
        return velocity * (sec_date - ref_date)

    else:
        if ref_date is None or sec_date is None:
            msg = "ref_date and sec_date are required for grid_type='variable'"
            raise ValueError(msg)

        cache_file = cache_dir / f"gnss_los_disp_{ref_date:.4f}_{sec_date:.4f}.npy"
        if cache_file.exists() and not recompute:
            logger.info("Loading cached GNSS LOS displacement from %s", cache_file)
            return np.load(cache_file)

        result = gnss_ref.compute_displacement_los(
            ref_date=ref_date,
            sec_date=sec_date,
            los_east=los_east,
            los_north=los_north,
            los_up=los_up,
            netcdf_file=netcdf_file,
            method="rbf",
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, result)
        logger.info("Saved GNSS LOS displacement to %s", cache_file)
        return result
