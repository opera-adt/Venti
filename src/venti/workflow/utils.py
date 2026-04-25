"""Utility functions for timeseries calibration.

This module provides helper functions for date handling, downsampling,
and other common operations.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def extract_dates_from_filename(
    filename: str | Path,
) -> tuple[date | None, date | None]:
    """Extract reference and secondary dates from OPERA filename.

    Parameters
    ----------
    filename : str or Path
        Filename containing dates in format YYYYMMDDTHHMMSS

    Returns
    -------
    tuple
        (reference_date, secondary_date) as datetime.date objects
        Returns (None, date) if only one date found

    Examples
    --------
    ::

        ref, sec = extract_dates_from_filename(
            'OPERA_L3_DISP-S1_20200101T000000_20200115T000000.nc'
        )

    """
    filename = str(filename)
    pattern = r"\d{8}T\d{6}"
    matches = re.findall(pattern, filename)

    start_date, end_date = None, None

    if len(matches) >= 2:
        start_date = datetime.strptime(matches[0], "%Y%m%dT%H%M%S").date()
        end_date = datetime.strptime(matches[1], "%Y%m%dT%H%M%S").date()
    elif len(matches) == 1:
        end_date = datetime.strptime(matches[0], "%Y%m%dT%H%M%S").date()

    return start_date, end_date


def datetime_to_decimal_year(dt: datetime) -> float:
    """Convert datetime to decimal year.

    Parameters
    ----------
    dt : datetime
        DateTime object to convert

    Returns
    -------
    float
        Decimal year (e.g., 2020.5 for middle of 2020)

    Examples
    --------
    ::

        dec_year = datetime_to_decimal_year(datetime(2020, 7, 1))
        # Returns approximately 2020.5

    """
    year_start = datetime(dt.year, 1, 1)
    next_year_start = datetime(dt.year + 1, 1, 1)
    year_length = (next_year_start - year_start).total_seconds()
    seconds_into_year = (dt - year_start).total_seconds()
    decimal_year = dt.year + seconds_into_year / year_length
    return round(decimal_year, 4)


def match_correction_to_displacement(
    correction_files: list[Path] | None, displacement_files: list[Path]
) -> list[tuple[Path | None, Path]]:
    """Match correction files to displacement files by date.

    Parameters
    ----------
    correction_files : list of Path or None
        List of correction files (e.g., tropospheric corrections).
        If ``None``, returns pairs with ``None`` for the correction file.
    displacement_files : list of Path
        List of displacement files

    Returns
    -------
    list of tuple
        List of ``(correction_file, displacement_file)`` pairs where
        ``correction_file`` is ``None`` when no corrections are available.

    Examples
    --------
    ::

        tropo_files = sorted(Path('tropo/').glob('*.tif'))
        disp_files = sorted(Path('disp/').glob('*.nc'))
        matches = match_correction_to_displacement(tropo_files, disp_files)

    """
    # Build dictionaries keyed by dates
    if correction_files is not None:
        corr_dict = {extract_dates_from_filename(f): f for f in correction_files}
    disp_dict = {extract_dates_from_filename(f): f for f in displacement_files}

    matches: list[tuple[Path | None, Path]] = []

    # If no corrections, pair with None
    if correction_files is None:
        for _dates, disp_file in disp_dict.items():
            matches.append((None, disp_file))
        return matches

    # Match by dates
    for dates, disp_file in disp_dict.items():
        # Prefer a combined differential file keyed by (ref_date, sec_date)
        if dates in corr_dict:
            matches.append((corr_dict[dates], disp_file))
        # Fall back to a single-epoch file keyed by (None, secondary_date)
        elif (None, dates[1]) in corr_dict:
            matches.append((corr_dict[(None, dates[1])], disp_file))
        else:
            logger.warning(f"No correction file matched for {dates}")

    return matches


def compute_average_temporal_coherence(
    netcdf_files: list[Path], output_dir: Path, variable: str = "temporal_coherence"
) -> Path:
    """Compute average temporal coherence from multiple NetCDF files.

    Parameters
    ----------
    netcdf_files : list of Path
        List of NetCDF files
    output_dir : Path
        Output directory
    variable : str, optional
        Variable name, by default 'temporal_coherence'

    Returns
    -------
    Path
        Path to output GeoTIFF with average temporal coherence

    Examples
    --------
    ::

        nc_files = sorted(Path('data/').glob('*.nc'))
        avg_coh = compute_average_temporal_coherence(nc_files, Path('output/'))

    """
    import xarray as xr

    output_file = output_dir / f"average_{variable}.tif"

    if output_file.exists():
        logger.info(f"Using existing average {variable}: {output_file}")
        return output_file

    logger.info(f"Computing average {variable} from {len(netcdf_files)} files")

    # Extract CRS from xarray dataset directly
    crs = None
    with xr.open_dataset(netcdf_files[0]) as ds:
        if "spatial_ref" in ds:
            sr = ds["spatial_ref"]
            for attr in ("crs_wkt", "spatial_ref", "wkt"):
                if attr in sr.attrs:
                    crs = sr.attrs[attr]
                    break
        elif "crs" in ds.attrs:
            crs = ds.attrs["crs"]

    if crs is None:
        logger.warning("Could not extract CRS from NetCDF")

    # Compute incremental mean to avoid loading all files into memory at once
    acc = None
    count = 0
    for f in netcdf_files:
        with xr.open_dataset(f) as ds:
            if variable not in ds:
                continue
            data = ds[variable].values
        if acc is None:
            acc = data.astype("float64")
        else:
            acc += data
        count += 1

    if acc is None or count == 0:
        msg = f"Variable '{variable}' not found in any files"
        raise ValueError(msg)

    average_da = xr.DataArray(
        (acc / count).astype("float32"),
        dims=["y", "x"],
    )

    if crs is not None:
        average_da = average_da.rio.write_crs(crs)
        average_da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    average_da.rio.to_raster(output_file)

    logger.info(f"Saved average {variable} to: {output_file}")
    return output_file


def parse_window_size_meters(window_meters: float, posting_meters: float = 30.0) -> int:
    """Convert window size from meters to pixels.

    Parameters
    ----------
    window_meters : float
        Window size in meters
    posting_meters : float, optional
        Pixel posting in meters, by default 30.0 (for OPERA)

    Returns
    -------
    int
        Window size in pixels

    Examples
    --------
    ::

        win_size = parse_window_size_meters(30000, posting_meters=30)
        # Returns 1000 pixels

    """
    window_pixels = int(np.rint(window_meters / posting_meters))
    return window_pixels


def get_file_dates(file_path: str | Path) -> tuple[float, float]:
    """Extract dates from file and convert to decimal years.

    Parameters
    ----------
    file_path : str or Path
        Path to file with dates in filename

    Returns
    -------
    tuple
        (ref_decimal_year, sec_decimal_year)

    Examples
    --------
    ::

        ref_year, sec_year = get_file_dates('OPERA_20200101T000000_20200115T000000.nc')

    """
    from opera_utils import get_dates

    dates = get_dates(str(file_path))
    ref_decimal = datetime_to_decimal_year(dates[0])
    sec_decimal = datetime_to_decimal_year(dates[1])

    return ref_decimal, sec_decimal


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, create if needed.

    Parameters
    ----------
    path : str or Path
        Directory path

    Returns
    -------
    Path
        Path object for directory

    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
