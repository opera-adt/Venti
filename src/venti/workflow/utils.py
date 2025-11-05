"""Utility functions for timeseries calibration.

This module provides helper functions for date handling, downsampling,
and other common operations.
"""

from __future__ import annotations

import numpy as np
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from scipy.ndimage import zoom
import logging

logger = logging.getLogger(__name__)


def extract_dates_from_filename(
    filename: str | Path
) -> tuple[Optional[datetime.date], Optional[datetime.date]]:
    """
    Extract reference and secondary dates from OPERA filename.

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
    """
    Convert datetime to decimal year.

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
    correction_files: Optional[list[Path]],
    displacement_files: list[Path]
) -> list[tuple[str, Path]]:
    """
    Match correction files to displacement files by date.

    Parameters
    ----------
    correction_files : list of Path or None
        List of correction files (e.g., tropospheric corrections)
        If None, returns matches with "None" for corrections
    displacement_files : list of Path
        List of displacement files

    Returns
    -------
    list of tuple
        List of (correction_file, displacement_file) pairs

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

    matches = []

    # If no corrections, pair with "None"
    if correction_files is None:
        for dates, disp_file in disp_dict.items():
            matches.append(("None", disp_file))
        return matches

    # Match by dates
    for dates, disp_file in disp_dict.items():
        # Try matching with (None, secondary_date)
        corr_date_key = (None, dates[1])
        if corr_date_key in corr_dict:
            matches.append((corr_dict[corr_date_key], disp_file))
        else:
            logger.warning(f"No correction file matched for {dates}")

    return matches


def downsample_array(
    array: np.ndarray,
    factor: int
) -> np.ndarray:
    """
    Downsample array by given factor using mean aggregation.

    Parameters
    ----------
    array : np.ndarray
        2D array to downsample
    factor : int
        Downsampling factor (e.g., 2 = half resolution)

    Returns
    -------
    np.ndarray
        Downsampled array

    Examples
    --------
    ::

        downsampled = downsample_array(data, factor=4)

    Notes
    -----
    NaN values are preserved. If any pixel in the downsampled region is NaN,
    the output pixel is NaN.
    """
    if factor == 1:
        return array

    # Preserve NaN values
    nan_mask = np.isnan(array)

    # Downsample mask - if ANY pixel is NaN, mark as NaN
    mask_downsampled = zoom(nan_mask.astype(float), 1.0 / factor, order=0) > 0.0

    # For valid data, use nanmean-like behavior
    array_filled = np.where(nan_mask, 0, array)

    # Downsample data
    downsampled = zoom(array_filled, 1.0 / factor, order=1)

    # Re-apply NaN mask
    downsampled[mask_downsampled] = np.nan

    logger.debug(
        f"Downsampled array from {array.shape} to {downsampled.shape} (factor={factor})"
    )

    return downsampled


def upsample_array(
    array: np.ndarray,
    target_shape: tuple[int, int]
) -> np.ndarray:
    """
    Upsample array to target shape using bilinear interpolation.

    Parameters
    ----------
    array : np.ndarray
        2D array to upsample
    target_shape : tuple
        Target shape (rows, cols)

    Returns
    -------
    np.ndarray
        Upsampled array

    Examples
    --------
    ::

        upsampled = upsample_array(downsampled_data, original_shape)

    Notes
    -----
    NaN values are preserved during upsampling.
    """
    if array.shape == target_shape:
        return array

    # Calculate zoom factors
    zoom_factors = (
        target_shape[0] / array.shape[0],
        target_shape[1] / array.shape[1]
    )

    # Preserve NaN values
    nan_mask = np.isnan(array)
    array_filled = np.where(nan_mask, 0, array)

    # Upsample using bilinear interpolation
    upsampled = zoom(array_filled, zoom_factors, order=1)

    # Upsample mask
    mask_upsampled = zoom(nan_mask.astype(float), zoom_factors, order=0) > 0.5

    # Re-apply NaN mask
    upsampled[mask_upsampled] = np.nan

    logger.debug(
        f"Upsampled array from {array.shape} to {upsampled.shape}"
    )

    return upsampled


def compute_average_temporal_coherence(
    netcdf_files: list[Path],
    output_dir: Path,
    variable: str = 'temporal_coherence'
) -> Path:
    """
    Compute average temporal coherence from multiple NetCDF files.

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
    import rioxarray

    output_file = output_dir / f"average_{variable}.tif"

    if output_file.exists():
        logger.info(f"Using existing average {variable}: {output_file}")
        return output_file

    logger.info(f"Computing average {variable} from {len(netcdf_files)} files")

    # Read first file to get CRS
    data_path = f"NETCDF:{netcdf_files[0]}:/{variable}"
    try:
        import rasterio as rio
        with rio.open(data_path) as src:
            crs = src.crs
    except Exception:
        crs = None
        logger.warning("Could not extract CRS from NetCDF")

    # Read and stack all files
    data_arrays = []
    for f in netcdf_files:
        ds = xr.open_dataset(f)
        if variable in ds:
            data_arrays.append(ds[variable])
        ds.close()

    if not data_arrays:
        raise ValueError(f"Variable '{variable}' not found in any files")

    # Compute mean
    stacked = xr.concat(data_arrays, dim="stack")
    average = stacked.mean(dim="stack", skipna=True)

    # Set CRS if available
    if crs is not None:
        average = average.rio.write_crs(crs)
        average.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    # Save to GeoTIFF
    output_dir.mkdir(parents=True, exist_ok=True)
    average.rio.to_raster(output_file)

    logger.info(f"Saved average {variable} to: {output_file}")
    return output_file


def parse_window_size_meters(
    window_meters: float,
    posting_meters: float = 30.0
) -> int:
    """
    Convert window size from meters to pixels.

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


def get_file_dates(
    file_path: str | Path
) -> tuple[float, float]:
    """
    Extract dates from file and convert to decimal years.

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
    """
    Ensure directory exists, create if needed.

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
