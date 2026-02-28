"""Utility functions for timeseries calibration.

This module provides helper functions for date handling, downsampling,
and other common operations.
"""

from __future__ import annotations

import logging
import re
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

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
        # Try matching with (None, secondary_date)
        corr_date_key = (None, dates[1])
        if corr_date_key in corr_dict:
            matches.append((corr_dict[corr_date_key], disp_file))
        else:
            logger.warning(f"No correction file matched for {dates}")

    return matches


def downsample_array(
    array: np.ndarray,
    factor: int,
    method: str = "mean",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Downsample array by given factor using specified aggregation method.

    Parameters
    ----------
    array : np.ndarray
        2D array to downsample
    factor : int
        Downsampling factor (e.g., 2 = half resolution)
    method : str, optional
        Aggregation method: 'mean' or 'median', by default 'mean'
    weights : np.ndarray, optional
        Weight array for weighted downsampling (same shape as array).
        If provided, computes weighted mean. Ignored if method='median'.

    Returns
    -------
    np.ndarray
        Downsampled array

    Examples
    --------
    ::

        # Simple mean downsampling
        downsampled = downsample_array(data, factor=4)

        # Median downsampling
        downsampled = downsample_array(data, factor=4, method='median')

        # Weighted mean downsampling
        downsampled = downsample_array(data, factor=4, method='mean', weights=coherence)

    Notes
    -----
    NaN values are handled using nanmean or nanmedian. Blocks with all NaN
    values will result in NaN in the output.

    """
    if factor == 1:
        return array

    if method not in ["mean", "median"]:
        msg = f"Invalid method '{method}'. Must be 'mean' or 'median'"
        raise ValueError(msg)

    # Calculate output shape
    new_shape = (array.shape[0] // factor, array.shape[1] // factor)

    # Trim array to be evenly divisible by factor
    trimmed_rows = new_shape[0] * factor
    trimmed_cols = new_shape[1] * factor
    array_trimmed = array[:trimmed_rows, :trimmed_cols]

    if weights is not None and method == "mean":
        weights_trimmed = weights[:trimmed_rows, :trimmed_cols]
        # Ensure weights are valid
        weights_trimmed = np.where(np.isnan(weights_trimmed), 0, weights_trimmed)
        weights_trimmed = np.where(weights_trimmed < 0, 0, weights_trimmed)

    # Reshape to blocks
    blocks = array_trimmed.reshape(
        new_shape[0], factor, new_shape[1], factor
    ).transpose(0, 2, 1, 3)

    if method == "mean":
        if weights is not None:
            # Weighted mean
            weight_blocks = weights_trimmed.reshape(
                new_shape[0], factor, new_shape[1], factor
            ).transpose(0, 2, 1, 3)

            # Compute weighted mean, handling NaN values
            with np.errstate(invalid="ignore", divide="ignore"):
                # Set NaN values to 0 weight
                weight_blocks_masked = np.where(np.isnan(blocks), 0, weight_blocks)
                data_masked = np.where(np.isnan(blocks), 0, blocks)

                # Sum of weighted values
                weighted_sum = np.sum(data_masked * weight_blocks_masked, axis=(2, 3))
                # Sum of weights
                weight_sum = np.sum(weight_blocks_masked, axis=(2, 3))

                # Weighted mean
                downsampled = weighted_sum / weight_sum

                # Set to NaN where all weights are zero
                downsampled = np.where(weight_sum == 0, np.nan, downsampled)
        else:
            # Regular mean, ignoring NaN
            # Suppress warning about mean of empty slice (when all values are NaN)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"Mean of empty slice")
                downsampled = np.nanmean(blocks, axis=(2, 3))
    else:  # median
        # Use nanmedian, ignoring NaN values
        # Suppress warning about invalid value encountered
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            downsampled = np.nanmedian(blocks, axis=(2, 3))

    logger.debug(
        f"Downsampled array from {array.shape} to {downsampled.shape} "
        f"(factor={factor}, method={method}, weighted={weights is not None})"
    )

    return downsampled


def upsample_array(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Upsample array to target shape using bilinear interpolation.

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
    zoom_factors = (target_shape[0] / array.shape[0], target_shape[1] / array.shape[1])

    # Preserve NaN values
    nan_mask = np.isnan(array)
    array_filled = np.where(nan_mask, 0, array)

    # Upsample using bilinear interpolation
    upsampled = zoom(array_filled, zoom_factors, order=1)

    # Upsample mask
    mask_upsampled = zoom(nan_mask.astype(float), zoom_factors, order=0) > 0.5

    # Re-apply NaN mask
    upsampled[mask_upsampled] = np.nan

    logger.debug(f"Upsampled array from {array.shape} to {upsampled.shape}")

    return upsampled


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
        msg = f"Variable '{variable}' not found in any files"
        raise ValueError(msg)

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
