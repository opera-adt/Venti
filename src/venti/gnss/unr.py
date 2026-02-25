"""UNR GNSS grid timeseries download and station velocity estimation.

Interfaces with the University of Nevada, Reno (UNR) geodesy database:
https://geodesy.unr.edu/grid_timeseries/
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import scipy.linalg
from numpy.linalg import lstsq
from shapely.geometry import Point, box

logger = logging.getLogger(__name__)

# UNR timeseries endpoints keyed by reference frame
_UNR_URLS: dict[str, dict[str, str]] = {
    "IGS20": {
        "grid": "https://geodesy.unr.edu/grid_timeseries/Version0.2/grid_latlon_lookup.txt",
        "data": "https://geodesy.unr.edu/grid_timeseries/Version0.2/time_variable_gridded/IGS20/",
    },
    "IGS14": {
        "grid": "https://geodesy.unr.edu/grid_timeseries/grid_latlon_lookup.txt",
        "data": "https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/IGS14/",
    },
}

# Column layout for UNR .tenv8 station files
_STATION_COLUMNS = ["year", "east", "north", "up", "sigma_e", "sigma_n", "sigma_u"]

# Epoch-displacement column names expected in GeoDataFrames returned by this module
_DISP_COLUMNS = ["deast", "dnorth", "dup", "dsigma_e", "dsigma_n", "dsigma_u"]


def download_grid_lookup(output_dir: Path, reference_frame: str = "IGS20") -> Path:
    """Download the UNR grid station lookup table.

    Parameters
    ----------
    output_dir : Path
        Directory to write the lookup file.
    reference_frame : str, optional
        GNSS reference frame, ``'IGS20'`` or ``'IGS14'``, by default ``'IGS20'``.

    Returns
    -------
    Path
        Path to the downloaded lookup file.

    Raises
    ------
    ValueError
        If `reference_frame` is not supported.
    RuntimeError
        If the download fails.

    """
    if reference_frame not in _UNR_URLS:
        msg = f"Unsupported reference frame '{reference_frame}'. Choose from {list(_UNR_URLS)}"
        raise ValueError(msg)

    url = _UNR_URLS[reference_frame]["grid"]
    output_path = output_dir / "grid_latlon_lookup.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading UNR grid lookup from %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")

    logger.info("Saved grid lookup to %s", output_path)
    return output_path


def find_stations_in_bounds(
    grid_lookup_path: Path,
    bounds_snwe: tuple[float, float, float, float],
    utm_epsg: int,
    padding: float = 0.0,
) -> gpd.GeoDataFrame:
    """Filter the UNR grid station table to those within a bounding box.

    Parameters
    ----------
    grid_lookup_path : Path
        Path to the UNR ``grid_latlon_lookup.txt`` file.
    bounds_snwe : tuple of float
        Bounding box as ``(south, north, west, east)`` in the UTM CRS.
    utm_epsg : int
        EPSG code of the UTM projection matching `bounds_snwe`.
    padding : float, optional
        Extra padding in the same units as `bounds_snwe`, by default ``0.0``.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of stations within the bounds, indexed by station ID,
        with columns ``lon``, ``lat``, and a ``geometry`` column in UTM.

    """
    df = pd.read_csv(
        grid_lookup_path,
        sep=r"\s+",
        header=None,
        names=["id", "lon", "lat"],
    )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    gdf_utm = gdf.to_crs(epsg=utm_epsg)
    gdf_utm = gdf_utm.set_index("id")

    S, N, W, E = bounds_snwe
    bbox = box(W - padding, S - padding, E + padding, N + padding)
    return gdf_utm[gdf_utm.within(bbox)]


def download_station(
    station_id: int,
    output_dir: Path,
    reference_frame: str = "IGS20",
) -> Path:
    """Download a single UNR GNSS grid station time series file.

    Parameters
    ----------
    station_id : int
        Numeric station ID (zero-padded to six digits in the filename).
    output_dir : Path
        Directory to write the station file.
    reference_frame : str, optional
        GNSS reference frame, by default ``'IGS20'``.

    Returns
    -------
    Path
        Path to the downloaded station file.

    Raises
    ------
    RuntimeError
        If the download fails.

    """
    filename = f"{station_id:06d}_{reference_frame}.tenv8"
    dest = output_dir / filename

    if dest.exists():
        logger.debug("Station file already exists: %s", dest)
        return dest

    url = _UNR_URLS[reference_frame]["data"] + filename
    logger.debug("Downloading station %s from %s", station_id, url)

    response = requests.get(url, timeout=60)
    if not response.ok:
        msg = f"Download failed for station {station_id} (HTTP {response.status_code})"
        raise RuntimeError(msg)

    dest.write_text(response.text, encoding="utf-8")
    return dest


def calculate_station_velocity(
    station_file: Path,
    start_year: float = 2014.0,
) -> tuple[float, float, float, float, float, float]:
    """Estimate east/north/up velocities via weighted linear regression.

    Fits a weighted least-squares model ``displacement = velocity * time + offset``
    to each component independently, using the inverse-variance weights.

    Parameters
    ----------
    station_file : Path
        UNR ``.tenv8`` station file (whitespace-delimited, columns ordered as
        ``year east north up sigma_e sigma_n sigma_u ...``).
    start_year : float, optional
        Exclude observations before this decimal year, by default ``2014.0``.

    Returns
    -------
    tuple of float
        ``(ve, vn, vu, sigma_ve, sigma_vn, sigma_vu)`` — velocities and their
        standard errors in the same units as the station file (typically m/yr).

    """
    data = np.loadtxt(station_file)

    t = data[:, 0]
    east, north, up = data[:, 1], data[:, 2], data[:, 3]
    sigma_e, sigma_n, sigma_u = data[:, 4], data[:, 5], data[:, 6]

    mask = t >= start_year
    t, east, north, up = t[mask], east[mask], north[mask], up[mask]
    sigma_e, sigma_n, sigma_u = sigma_e[mask], sigma_n[mask], sigma_u[mask]

    # Design matrix [time, 1] for velocity + offset
    X = np.column_stack([t, np.ones_like(t)])

    def _wls(y: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
        """Weighted least-squares slope and its standard error."""
        w = 1.0 / sigma**2
        W = np.diag(w)
        XtW = X.T @ W
        beta, _, _, _ = lstsq(XtW @ X, XtW @ y, rcond=None)

        residuals = y - X @ beta
        dof = max(len(t) - 2, 1)
        rss = np.sum(w * residuals**2)
        cov = np.linalg.inv(XtW @ X)
        slope_std = np.sqrt(rss / dof * cov[0, 0])
        return float(beta[0]), float(slope_std)

    ve, ve_std = _wls(east, sigma_e)
    vn, vn_std = _wls(north, sigma_n)
    vu, vu_std = _wls(up, sigma_u)

    return ve, vn, vu, ve_std, vn_std, vu_std


def read_epoch_displacements(
    station_files: list[Path],
    ref_date: float,
    sec_date: float,
    station_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Read GNSS displacements for a specific epoch pair from station files.

    Finds the observation nearest to each date and computes the difference
    ``ref - sec`` in east/north/up.

    Parameters
    ----------
    station_files : list of Path
        UNR station files to read.
    ref_date : float
        Reference epoch as decimal year.
    sec_date : float
        Secondary epoch as decimal year.
    station_gdf : gpd.GeoDataFrame
        GeoDataFrame of station metadata (geometry in UTM), indexed by station ID.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with columns ``deast``, ``dnorth``, ``dup``,
        ``dsigma_e``, ``dsigma_n``, ``dsigma_u``, and a ``geometry`` column.

    """
    rows = []
    for path in station_files:
        station_id = int(path.name.split("_")[0])
        df = pd.read_csv(path, sep=r"\s+", names=_STATION_COLUMNS, usecols=range(7))

        ref_row = df.iloc[(df["year"] - ref_date).abs().argmin()]
        sec_row = df.iloc[(df["year"] - sec_date).abs().argmin()]

        rows.append({
            "id": station_id,
            "deast": ref_row["east"] - sec_row["east"],
            "dnorth": ref_row["north"] - sec_row["north"],
            "dup": ref_row["up"] - sec_row["up"],
            "dsigma_e": ref_row["sigma_e"] - sec_row["sigma_e"],
            "dsigma_n": ref_row["sigma_n"] - sec_row["sigma_n"],
            "dsigma_u": ref_row["sigma_u"] - sec_row["sigma_u"],
        })

    diff_df = pd.DataFrame(rows).set_index("id")
    return diff_df.join(station_gdf[["geometry"]], how="inner")
