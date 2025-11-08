"""Tropospheric correction download and processing for DISP-S1 products.

This module provides tools to download, crop, and apply tropospheric corrections
from HRRR weather model data to DISP-S1 displacement products.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import asf_search as asf
import geopandas as gpd
import numpy as np
import requests
import rioxarray as rxr
import xarray as xr
from opera_utils import get_frame_bbox, get_frame_geojson
from opera_utils.tropo import apply_tropo, crop_tropo
from shapely.geometry import shape
from tqdm import tqdm

# Constants
ASF_TROPO_COLLECTION = "C3717139408-ASF"
BURST_DB_URL = (
    "https://github.com/opera-adt/burst_db/releases/download/v0.13.0/"
    "opera-disp-s1-consistent-burst-ids-2025-09-16-2016-07-01_to_2024-12-31.json"
)
DATE_FORMATS = ("%Y-%m-%d", "%Y%m%d")
DEFAULT_OUTPUT_DIR = Path("./tropo")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Suppress verbose logging from third-party libraries
logging.getLogger("asf_search").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_date(date_str: str | None) -> datetime | None:
    """Parse date string into datetime object.

    Supports both ISO format (YYYY-MM-DD) and compact format (YYYYMMDD).

    Parameters
    ----------
    date_str : str or None
        Date string to parse, or None.

    Returns
    -------
    datetime or None
        Parsed datetime object, or None if input is None.

    Raises
    ------
    ValueError
        If date string format is invalid.

    Examples
    --------
    >>> parse_date("2024-01-15")
    datetime.datetime(2024, 1, 15, 0, 0)
    >>> parse_date("20240115")
    datetime.datetime(2024, 1, 15, 0, 0)

    """
    if date_str is None:
        return None

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    msg = (
        f"Invalid date format: {date_str}. Expected formats: {', '.join(DATE_FORMATS)}"
    )
    raise ValueError(msg)


def load_burst_database(url: str = BURST_DB_URL) -> gpd.GeoDataFrame:
    """Load OPERA burst database JSON to GeoDataFrame.

    Parameters
    ----------
    url : str, optional
        URL to the burst database JSON file.
        Default is the OPERA consistent burst database.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with burst information including burst_id, geometry,
        and sensing times.

    Raises
    ------
    requests.HTTPError
        If the database cannot be downloaded.

    Examples
    --------
    >>> gdf = load_burst_database()
    >>> print(gdf.columns)
    Index(['burst_id', 'geometry', 'sensing_time_start', ...])

    """
    logger.info("Loading OPERA consistent burst database")

    resp = requests.get(url)
    resp.raise_for_status()

    data = resp.json()
    logger.info(f"Database generation date: {data['metadata']['generation_time']}")

    features = []
    for burst_id, info in data["data"].items():
        features.append(
            {
                "burst_id": burst_id,
                "geometry": shape(info["geometry"]) if "geometry" in info else None,
                **{k: v for k, v in info.items() if k != "geometry"},
            }
        )

    return gpd.GeoDataFrame(features, crs="EPSG:4326")


def extract_sensing_times_from_file(disp_file: Path) -> list[datetime]:
    """Extract sensing times from a DISP-S1 NetCDF file or filename.

    Parses the DISP-S1 filename to extract reference and secondary dates.
    Filename format:
    OPERA_L3_DISP-S1_IW_F{frame}_VV_{ref_date}_{sec_date}_v{version}_{prod_date}.nc

    Parameters
    ----------
    disp_file : Path
        Path to DISP-S1 NetCDF file.

    Returns
    -------
    list[datetime]
        List of unique sensing times from reference and secondary dates.

    Raises
    ------
    FileNotFoundError
        If DISP file does not exist.
    ValueError
        If dates cannot be parsed from filename or file.

    Examples
    --------
    >>> # Filename: OPERA_L3_DISP-S1_IW_F08887_VV_{dateTtime}Z_{dateTtime}Z_v1.0.nc
    >>> times = extract_sensing_times_from_file(Path("OPERA_L3_DISP-S1_...nc"))
    >>> print(len(times))
    2
    >>> print(times[0])
    2016-07-05 00:28:09

    """
    if not disp_file.exists():
        msg = f"DISP file not found: {disp_file}"
        raise FileNotFoundError(msg)

    logger.info(f"Extracting sensing times from {disp_file.name}")

    filename = disp_file.name

    # Split by underscore and find the datetime strings
    parts = filename.split("_")

    # Find parts that match datetime format (YYYYMMDDTHHMMSSZ)
    datetime_parts = [p for p in parts if len(p) == 16 and p.endswith("Z") and "T" in p]

    if len(datetime_parts) < 2:
        msg = (
            f"Cannot parse reference and secondary dates from filename: {filename}. "
            "Expected format:"
            " OPERA_L3_DISP-S1_IW_F{frame}_VV_{ref_date}_{sec_date}_"
            "v{version}_{prod_date}.nc"
        )
        raise ValueError(msg)

    # First datetime is reference, second is secondary
    ref_date_str = datetime_parts[0]
    sec_date_str = datetime_parts[1]

    try:
        # Parse ISO format with Z suffix: 20160705T002809Z
        ref_date = datetime.strptime(ref_date_str, "%Y%m%dT%H%M%SZ")
        sec_date = datetime.strptime(sec_date_str, "%Y%m%dT%H%M%SZ")
    except ValueError as e:
        msg = f"Failed to parse dates from filename {filename}: {e}"
        raise ValueError(msg) from e

    sensing_times = sorted({ref_date, sec_date})
    logger.info(
        f"Parsed sensing times: {ref_date.isoformat()} (ref), "
        f"{sec_date.isoformat()} (sec)"
    )

    return sensing_times


def extract_frame_id_from_filename(disp_file: Path) -> int:
    """Extract frame ID from a DISP-S1 filename.

    Parses the DISP-S1 filename to extract the frame identifier.
    Filename format:
    OPERA_L3_DISP-S1_IW_F{frame}_VV_{ref_date}_{sec_date}_v{version}_{prod_date}.nc

    Parameters
    ----------
    disp_file : Path
        Path to DISP-S1 NetCDF file.

    Returns
    -------
    int
        Frame identifier.

    Raises
    ------
    ValueError
        If frame ID cannot be parsed from filename.

    Examples
    --------
    >>> # Filename:
    OPERA_L3_DISP-S1_IW_F08887_VV_20160705T002809Z_20161208T002812Z_v1.0.nc
    >>> frame_id = extract_frame_id_from_filename(Path("OPERA_L3_DISP-S1_...nc"))
    >>> print(frame_id)
    8887

    """
    filename = disp_file.name

    # Find the frame ID part (starts with F followed by digits)
    parts = filename.split("_")
    frame_parts = [p for p in parts if p.startswith("F") and p[1:].isdigit()]

    if not frame_parts:
        msg = (
            f"Cannot parse frame ID from filename: {filename}. "
            "Expected format: OPERA_L3_DISP-S1_IW_F{frame}_VV_..."
        )
        raise ValueError(msg)

    # Remove leading 'F' and convert to int
    frame_str = frame_parts[0][1:]
    return int(frame_str)


def extract_sensing_times_from_database(
    frame_id: int,
    burst_db: gpd.GeoDataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[datetime]:
    """Extract sensing times for a frame from the burst database.

    Parameters
    ----------
    frame_id : int
        OPERA frame identifier.
    burst_db : gpd.GeoDataFrame
        Burst database GeoDataFrame.
    start : datetime or None, optional
        Start date for filtering. Default is None (no start limit).
    end : datetime or None, optional
        End date for filtering. Default is None (no end limit).

    Returns
    -------
    list[datetime]
        List of unique sensing times for the frame within the date range.

    Raises
    ------
    ValueError
        If no bursts found for the frame or no sensing times in date range.

    Examples
    --------
    >>> burst_db = load_burst_database()
    >>> times = extract_sensing_times_from_database(8887, burst_db)
    >>> print(len(times))
    150

    """
    logger.info(f"Extracting sensing times for frame {frame_id}")

    # Filter bursts for this frame
    frame_bursts = burst_db[burst_db["frame_id"] == frame_id]

    if frame_bursts.empty:
        msg = f"No bursts found for frame {frame_id} in database"
        raise ValueError(msg)

    sensing_times = []
    for _, burst in frame_bursts.iterrows():
        if "sensing_time_list" in burst:
            times = burst["sensing_time_list"]
            if times:
                sensing_times.extend(times)

    if not sensing_times:
        msg = f"No sensing times found for frame {frame_id} in database"
        raise ValueError(msg)

    # Convert to datetime
    unique_times = sorted({datetime.fromisoformat(t) for t in sensing_times})

    # Filter by date range
    if start:
        unique_times = [t for t in unique_times if t >= start]
    if end:
        unique_times = [t for t in unique_times if t <= end]

    if not unique_times:
        msg = (
            f"No sensing times found for frame {frame_id} "
            f"in date range [{start}, {end}]"
        )
        raise ValueError(msg)

    logger.info(f"Found {len(unique_times)} sensing times in date range")
    return unique_times


def download_tropo_urls(output_dir: Path) -> Path:
    """Search and download list of tropospheric correction URLs.

    Parameters
    ----------
    output_dir : Path
        Directory to save the URL list.

    Returns
    -------
    Path
        Path to the saved tropo_urls.txt file.

    Examples
    --------
    >>> url_file = download_tropo_urls(Path("./tropo"))
    >>> print(url_file)
    ./tropo/tropo_urls.txt

    """
    logger.info("Searching ASF for tropospheric corrections")

    results = asf.search(collections=[ASF_TROPO_COLLECTION])

    if not results:
        msg = (
            "No tropospheric correction products found in collection"
            f" {ASF_TROPO_COLLECTION}"
        )
        raise ValueError(msg)

    logger.info(f"Found {len(results)} tropospheric correction products")

    tropo_gdf = gpd.GeoDataFrame.from_features(results.geojson())

    if "url" not in tropo_gdf.columns:
        msg = "URL column not found in search results"
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    url_file = output_dir / "tropo_urls.txt"

    tropo_gdf["url"].to_csv(url_file, header=False, index=False)
    logger.info(f"Saved {len(tropo_gdf)} URLs to {url_file}")

    return url_file


def download_prep_tropo(
    sensing_times: list[datetime],
    frame_bounds: tuple[float, float, float, float],
    dem_path: Path,
    incidence_angle_path: Path,
    output_dir: Path,
    num_workers: int = 2,
    dem_hgt_buffer: float = 3000.0,
) -> None:
    """Download and prepare tropospheric correction data.

    Parameters
    ----------
    sensing_times : list[datetime]
        List of sensing times to process.
    frame_bounds : tuple[float, float, float, float]
        Geographic bounds (minx, miny, maxx, maxy).
    dem_path : Path
        Path to DEM file.
    incidence_angle_path : Path
        Path to incidence angle file.
    output_dir : Path
        Output directory for all products.
    num_workers : int, optional
        Number of parallel workers. Default is 2.
    dem_hgt_buffer : float, optional
        Height buffer above DEM max (meters). Default is 3000.0.

    Raises
    ------
    FileNotFoundError
        If DEM or incidence angle file does not exist.

    """
    if not dem_path.exists():
        msg = f"DEM file not found: {dem_path}"
        raise FileNotFoundError(msg)

    if not incidence_angle_path.exists():
        msg = f"Incidence angle file not found: {incidence_angle_path}"
        raise FileNotFoundError(msg)

    # Download URL list
    url_file = download_tropo_urls(output_dir)

    # Crop to frame and times
    logger.info(f"Cropping tropospheric data for {len(sensing_times)} sensing times")
    cropped_dir = output_dir / "cropped_tropo"
    cropped_dir.mkdir(parents=True, exist_ok=True)

    # Get max height from DEM with buffer for tropo download/crop
    with rxr.open_rasterio(dem_path) as hgt:
        max_dem_hgt = float(hgt.squeeze().max().values)
        hgt_max = max_dem_hgt + dem_hgt_buffer

    crop_tropo(
        url_file,
        sensing_times,
        frame_bounds,
        height_max=hgt_max,
        output_dir=cropped_dir,
        num_workers=num_workers,
    )
    cropped_files = sorted(cropped_dir.glob("tropo*.nc"))
    logger.info(f"Generated {len(cropped_files)} cropped files")

    # Apply corrections
    corrections_dir = output_dir / "tropo_corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)

    apply_tropo(
        cropped_files,
        dem_path=dem_path,
        incidence_angle_path=incidence_angle_path,
        output_dir=corrections_dir,
        subtract_first_date=False,  # Skip, do it later
        num_workers=num_workers,
    )

    logger.info(f"Corrections saved to {corrections_dir}")


def _reproject_to_disp_utm(input_file: Path, disp_file: Path) -> Path:
    """Reproject tropospheric correction to match DISP-S1 UTM grid.

    Parameters
    ----------
    input_file : Path
        Path to input tropospheric correction file.
    disp_file : Path
        Path to DISP-S1 file to match projection.

    Returns
    -------
    Path
        Path to reprojected output file.

    """
    with rxr.open_rasterio(input_file) as src, rxr.open_rasterio(disp_file) as disp:
        out = src.rio.reproject_match(disp)
        epsg = disp.rio.crs.to_epsg()
    out = out.squeeze(drop=True)
    output_dir = input_file.parents[1] / f"tropo_corrections_{epsg}"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{input_file.stem}_{epsg}.tif"
    output_path = output_dir / filename

    out.rio.to_raster(output_path)

    return output_path


def _reproject_to_epsg(input_file: Path, epsg: int) -> Path:
    """Reproject tropospheric correction to specified EPSG code.

    Parameters
    ----------
    input_file : Path
        Path to input tropospheric correction file.
    epsg : int
        Target EPSG code.

    Returns
    -------
    Path
        Path to reprojected output file.

    """
    with rxr.open_rasterio(input_file) as src:
        out = src.rio.reproject(f"EPSG:{epsg}")

    output_dir = input_file.parents[1] / f"tropo_corrections_{epsg}"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{input_file.stem}_{epsg}.tif"
    output_path = output_dir / filename

    out.rio.to_raster(output_path)

    return output_path


def _reproject_files_parallel(
    tropo_files: list[Path],
    reproject_fn: Callable,
    reproject_args: tuple,
    num_workers: int = 2,
) -> list[Path]:
    """Reproject multiple files in parallel.

    Parameters
    ----------
    tropo_files : list[Path]
        List of tropospheric correction files to reproject.
    reproject_fn : Callable
        Reprojection function to apply.
    reproject_args : tuple
        Additional arguments to pass to reprojection function.
    num_workers : int, optional
        Number of parallel workers. Default is 2.

    Returns
    -------
    list[Path]
        List of reprojected output file paths.

    """
    output_paths = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(reproject_fn, file, *reproject_args): file
            for file in tropo_files
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Reprojecting files",
            unit="file",
        ):
            input_file = futures[future]
            try:
                output_path = future.result()
                output_paths.append(output_path)
            except Exception:
                logger.exception(f"Failed to reproject {input_file}")
                raise
    logger.info(f"Corrections saved to {Path(output_paths[0]).parent}")
    return sorted(output_paths)


def process_tropo_from_file(
    disp_file: Path,
    dem_path: Path,
    incidence_angle_path: Path,
    output_dir: Path,
    num_workers: int = 2,
    dem_hgt_buffer: float = 3000.0,
    to_disp_epsg: bool = True,
) -> None:
    """Process tropospheric corrections using a DISP-S1 file.

    Extracts sensing times from the DISP file, downloads and crops
    tropospheric data, then applies corrections.

    Parameters
    ----------
    disp_file : Path
        Path to DISP-S1 NetCDF file.
    dem_path : Path
        Path to DEM file.
    incidence_angle_path : Path
        Path to incidence angle file.
    output_dir : Path
        Output directory for all products.
    num_workers : int, optional
        Number of parallel workers. Default is 2.
    dem_hgt_buffer : float, optional
        Height buffer above DEM max (meters). Default is 3000.0.
    to_disp_epsg : bool, optional
        Reproject to DISP file's UTM projection. Default is True.

    Examples
    --------
    >>> process_tropo_from_file(
    ...     disp_file=Path("DISP_frame_8887.nc"),
    ...     dem_path=Path("dem.tif"),
    ...     incidence_angle_path=Path("incidence.tif"),
    ...     output_dir=Path("./tropo")
    ... )

    """
    logger.info(f"Processing tropospheric corrections from file: {disp_file}")

    sensing_times = extract_sensing_times_from_file(disp_file)
    frame_id = extract_frame_id_from_filename(disp_file)

    selected_frame = get_frame_geojson([frame_id], as_geodataframe=True)
    frame_bounds = tuple(selected_frame.bounds.values[0])

    download_prep_tropo(
        sensing_times,
        frame_bounds,
        dem_path,
        incidence_angle_path,
        output_dir,
        num_workers,
        dem_hgt_buffer,
    )

    if to_disp_epsg:
        tropo_files = list((output_dir / "tropo_corrections").glob("*.tif"))
        logger.info(f"Reprojecting {len(tropo_files)} files to DISP grid in parallel")
        _reproject_files_parallel(
            tropo_files, _reproject_to_disp_utm, (disp_file,), num_workers
        )


def process_tropo_from_stack(
    disp_dir: Path,
    dem_path: Path,
    incidence_angle_path: Path,
    output_dir: Path,
    num_workers: int = 2,
    dem_hgt_buffer: float = 3000.0,
    frame_id: int | None = None,
    to_disp_epsg: bool = True,
) -> None:
    """Process tropospheric corrections from a directory of DISP-S1 files.

    Extracts sensing times from all DISP files in the directory,
    downloads and crops tropospheric data, then applies corrections.

    Parameters
    ----------
    disp_dir : Path
        Path to directory containing DISP-S1 NetCDF files.
    dem_path : Path
        Path to DEM file.
    incidence_angle_path : Path
        Path to incidence angle file.
    output_dir : Path
        Output directory for all products.
    num_workers : int, optional
        Number of parallel workers. Default is 2.
    dem_hgt_buffer : float, optional
        Height buffer above DEM max (meters). Default is 3000.0.
    frame_id : int or None, optional
        Specific frame ID to process. If None, processes all frames.
        Default is None.
    to_disp_epsg : bool, optional
        Reproject to DISP file's UTM projection. Default is True.

    Raises
    ------
    ValueError
        If multiple frame IDs are detected without specifying which to process.

    Examples
    --------
    >>> process_tropo_from_stack(
    ...     disp_dir=Path("./disp_files"),
    ...     dem_path=Path("dem.tif"),
    ...     incidence_angle_path=Path("incidence.tif"),
    ...     output_dir=Path("./tropo"),
    ...     frame_id=8887
    ... )

    """
    logger.info(f"Processing tropospheric corrections from dir: {disp_dir}")
    disp_list = list(Path(disp_dir).glob("OPERA_L3_DISP-S1*.nc"))
    logger.info(f"Found {len(disp_list)} DISP-S1 files in {disp_dir}")

    frame_id_list = [extract_frame_id_from_filename(f) for f in disp_list]
    unique_frame_ids = np.unique(frame_id_list)

    # Filter by frame_id if specified
    if frame_id is not None:
        disp_list = [
            f
            for f, fid in zip(disp_list, frame_id_list, strict=False)
            if fid == frame_id
        ]
        target_frame_id = frame_id
    else:
        if len(unique_frame_ids) > 1:
            msg = (
                f"Multiple frame IDs detected in {disp_dir}: {unique_frame_ids}. "
                "Rerun with specified frame_id parameter."
            )
            raise ValueError(msg)
        target_frame_id = int(unique_frame_ids[0])

    # Extract and flatten sensing times
    all_sensing_times = []
    for file in disp_list:
        times = extract_sensing_times_from_file(file)
        if isinstance(times, list):
            all_sensing_times.extend(times)
        else:
            all_sensing_times.append(times)
    sensing_times = np.unique(all_sensing_times).tolist()

    selected_frame = get_frame_geojson([target_frame_id], as_geodataframe=True)
    frame_bounds = tuple(selected_frame.bounds.values[0])

    download_prep_tropo(
        sensing_times,
        frame_bounds,
        dem_path,
        incidence_angle_path,
        output_dir,
        num_workers,
        dem_hgt_buffer,
    )

    if to_disp_epsg:
        tropo_files = list((output_dir / "tropo_corrections").glob("*.tif"))
        logger.info(f"Reprojecting {len(tropo_files)} files to DISP grid in parallel")
        _reproject_files_parallel(
            tropo_files, _reproject_to_disp_utm, (disp_list[0],), num_workers
        )


def process_tropo_from_db(
    frame_id: int,
    dem_path: Path,
    incidence_angle_path: Path,
    output_dir: Path,
    num_workers: int = 2,
    dem_hgt_buffer: float = 3000.0,
    start: datetime | None = None,
    end: datetime | None = None,
    to_disp_epsg: bool = True,
) -> None:
    """Process tropospheric corrections using frame ID and date range.

    Queries the OPERA consistent burst database for sensing times,
    downloads and crops tropospheric data, then applies corrections.

    Parameters
    ----------
    frame_id : int
        OPERA frame identifier.
    dem_path : Path
        Path to DEM file.
    incidence_angle_path : Path
        Path to incidence angle file.
    output_dir : Path
        Output directory for all products.
    num_workers : int, optional
        Number of parallel workers. Default is 2.
    dem_hgt_buffer : float, optional
        Height buffer above DEM max (meters). Default is 3000.0.
    start : datetime or None, optional
        Start date for filtering. Default is None (no start limit).
    end : datetime or None, optional
        End date for filtering. Default is None (no end limit).
    to_disp_epsg : bool, optional
        Reproject to frame's native EPSG. Default is True.

    Examples
    --------
    >>> process_tropo_from_db(
    ...     frame_id=8887,
    ...     dem_path=Path("dem.tif"),
    ...     incidence_angle_path=Path("incidence.tif"),
    ...     output_dir=Path("./tropo"),
    ...     start=datetime(2024, 1, 1),
    ...     end=datetime(2024, 12, 31)
    ... )

    """
    logger.info(
        f"Processing tropospheric corrections for frame {frame_id} "
        f"from {start or 'beginning'} to {end or 'present'}"
    )

    burst_db = load_burst_database()
    sensing_times = extract_sensing_times_from_database(frame_id, burst_db, start, end)

    selected_frame = get_frame_geojson([frame_id], as_geodataframe=True)
    frame_bounds = tuple(selected_frame.bounds.values[0])
    epsg = get_frame_bbox(frame_id)[0]

    download_prep_tropo(
        sensing_times,
        frame_bounds,
        dem_path,
        incidence_angle_path,
        output_dir,
        num_workers,
        dem_hgt_buffer,
    )

    if to_disp_epsg:
        tropo_files = list((output_dir / "tropo_corrections").glob("*.tif"))
        logger.info(f"Reprojecting {len(tropo_files)} files to EPSG:{epsg} in parallel")
        _reproject_files_parallel(tropo_files, _reproject_to_epsg, (epsg,), num_workers)


@dataclass
class FromFile:
    """Process tropospheric corrections from a DISP-S1 file.

    Attributes
    ----------
    disp_file : Path
        Path to DISP-S1 NetCDF file containing reference and secondary dates.
    dem_path : Path
        Path to DEM file for the frame.
    incidence_angle_path : Path
        Path to incidence angle file for the frame.
    output_dir : Path
        Output directory for tropospheric products.
    num_workers : int, optional
        Number of parallel workers. Default is 2.
    dem_hgt_buffer : float, optional
        Height buffer above DEM max (meters). Default is 3000.0.
    to_disp_epsg : bool, optional
        Reproject to DISP file's UTM projection. Default is True.

    """

    disp_file: Path
    dem_path: Path
    incidence_angle_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    num_workers: int = 2
    dem_hgt_buffer: float = 3000.0
    to_disp_epsg: bool = True

    def __call__(self) -> None:
        """Execute file-based tropospheric correction processing."""
        process_tropo_from_file(
            self.disp_file,
            self.dem_path,
            self.incidence_angle_path,
            self.output_dir,
            self.num_workers,
            self.dem_hgt_buffer,
            self.to_disp_epsg,
        )


@dataclass
class FromDir:
    """Process tropospheric corrections from directory with DISP-S1 files.

    Attributes
    ----------
    disp_dir : Path
        Path to directory containing DISP-S1 NetCDF files.
    dem_path : Path
        Path to DEM file for the frame.
    incidence_angle_path : Path
        Path to incidence angle file for the frame.
    output_dir : Path
        Output directory for tropospheric products.
    num_workers : int, optional
        Number of parallel workers. Default is 2.
    dem_hgt_buffer : float, optional
        Height buffer above DEM max (meters). Default is 3000.0.
    frame_id : int or None, optional
        Specific frame ID to process. If None, processes all frames.
        Default is None.
    to_disp_epsg : bool, optional
        Reproject to DISP file's UTM projection. Default is True.

    """

    disp_dir: Path
    dem_path: Path
    incidence_angle_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    num_workers: int = 2
    dem_hgt_buffer: float = 3000.0
    frame_id: int | None = None
    to_disp_epsg: bool = True

    def __call__(self) -> None:
        """Execute directory-based tropospheric correction processing."""
        process_tropo_from_stack(
            self.disp_dir,
            self.dem_path,
            self.incidence_angle_path,
            self.output_dir,
            self.num_workers,
            self.dem_hgt_buffer,
            self.frame_id,
            self.to_disp_epsg,
        )


@dataclass
class FromDB:
    """Process tropospheric corrections from frame ID and date range.

    Attributes
    ----------
    frame_id : int
        OPERA frame identifier.
    dem_path : Path
        Path to DEM file for the frame.
    incidence_angle_path : Path
        Path to incidence angle file for the frame.
    output_dir : Path
        Output directory for tropospheric products.
    start : str or None, optional
        Start date (YYYY-MM-DD or YYYYMMDD format).
    end : str or None, optional
        End date (YYYY-MM-DD or YYYYMMDD format).
    num_workers : int, optional
        Number of parallel workers. Default is 2.
    dem_hgt_buffer : float, optional
        Height buffer above DEM max (meters). Default is 3000.0.
    to_disp_epsg : bool, optional
        Reproject to frame's native EPSG. Default is True.

    """

    frame_id: int
    dem_path: Path
    incidence_angle_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    start: str | None = None
    end: str | None = None
    num_workers: int = 2
    dem_hgt_buffer: float = 3000.0
    to_disp_epsg: bool = True

    def __call__(self) -> None:
        """Execute database-based tropospheric correction processing."""
        process_tropo_from_db(
            self.frame_id,
            self.dem_path,
            self.incidence_angle_path,
            self.output_dir,
            num_workers=self.num_workers,
            dem_hgt_buffer=self.dem_hgt_buffer,
            start=parse_date(self.start),
            end=parse_date(self.end),
            to_disp_epsg=self.to_disp_epsg,
        )


def main() -> None:
    """Run CLI with file and stack subcommands."""
    import tyro

    tyro.extras.subcommand_cli_from_dict(
        {"db": FromDB, "file": FromFile, "stack": FromDir}
    )()


if __name__ == "__main__":
    main()
