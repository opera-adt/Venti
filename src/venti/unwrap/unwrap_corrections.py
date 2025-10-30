"""
Unwrap error correction module for interferometric phase unwrapping.

This module provides classes and functions to correct residual offsets in unwrapped
interferograms using watershed segmentation and regional median corrections.
"""
from __future__ import annotations

import numpy as np
import numba
import logging
from pathlib import Path
from skimage import filters, measure, segmentation
from scipy import ndimage as ndi
from typing import Optional, Tuple, Union
import xarray as xr

# Optional imports for GeoTIFF support
try:
    import rasterio as rio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    Affine = None

# Set up logger
logger = logging.getLogger(__name__)


@numba.njit(parallel=True)
def _update_disp_wavelength(input_disp, labeled_regions, valid_labels, unwrap_cycles, wavelength, disp_updated):
    """
    Update displacement values by removing wavelength cycle offsets.

    This function is optimized with Numba for parallel execution.

    Parameters
    ----------
    input_disp : np.ndarray
        Input displacement array
    labeled_regions : np.ndarray
        Array with labeled regions from watershed segmentation
    valid_labels : np.ndarray
        Array of valid region labels
    unwrap_cycles : np.ndarray
        Number of wavelength cycles to remove from each region
    wavelength : float
        The wavelength value for unwrap correction
    disp_updated : np.ndarray
        Output array to store corrected displacement
    """
    for idx in numba.prange(len(valid_labels)):
        label = valid_labels[idx]
        cycles = unwrap_cycles[idx]
        correction = cycles * wavelength
        rows, cols = np.where(labeled_regions == label)
        for i in range(len(rows)):
            r, c = rows[i], cols[i]
            disp_updated[r, c] = input_disp[r, c] - correction


class UnwrapCorrector:
    """
    Class for correcting unwrapping errors in interferometric phase data.

    This class implements watershed-based segmentation to identify coherent
    regions and correct unwrap errors by detecting and removing integer
    multiples of the wavelength.

    Parameters
    ----------
    min_region_area : int, optional
        Minimum area (in pixels) for valid regions, by default 20
    wavelength : float, optional
        The wavelength value for unwrap correction, by default 0.0555m for Sentinel-1.
        For phase in radians, use 2π. For other satellites, use appropriate wavelength.

    Attributes
    ----------
    min_region_area : int
        Minimum area threshold for region filtering
    wavelength : float
        Wavelength used for unwrap correction
    labeled_regions_ : np.ndarray or None
        Labeled regions from watershed segmentation (set after correction)
    valid_labels_ : np.ndarray or None
        Valid region labels after area filtering (set after correction)
    medians_ : np.ndarray or None
        Median displacement values for each region (set after correction)
    unwrap_cycles_ : np.ndarray or None
        Number of wavelength cycles corrected for each region (set after correction)
    n_regions_ : int or None
        Number of valid regions identified (set after correction)

    Examples
    --------
    Correct Sentinel-1 displacement (meters) - default::

        corrector = UnwrapCorrector()
        corrected = corrector.correct(disp, mask)
        info = corrector.get_region_info()
        print(f"Corrected {info['n_regions']} regions with cycles: {info['unwrap_cycles']}")

    Correct phase data (radians)::

        corrector = UnwrapCorrector(wavelength=2*np.pi)
        corrected = corrector.correct(phase, mask)
    """

    def __init__(self, min_region_area: int = 20, wavelength: float = 0.0555):
        """Initialize the UnwrapCorrector.

        Parameters
        ----------
        min_region_area : int, optional
            Minimum area (in pixels) for valid regions, by default 20
        wavelength : float, optional
            The wavelength value for unwrap correction (default is 0.0555m for Sentinel-1).
            For phase in radians, use 2π. Common values:
            - Sentinel-1 (C-band): 0.0555 m
            - ALOS-2 (L-band): 0.236 m
        """
        self.min_region_area = min_region_area
        self.wavelength = wavelength
        self.labeled_regions_ = None
        self.valid_labels_ = None
        self.medians_ = None
        self.unwrap_cycles_ = None
        self.n_regions_ = None

    def _prepare_displacement(
        self,
        input_disp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare displacement data for watershed segmentation.

        Parameters
        ----------
        input_disp : np.ndarray
            Input displacement field

        Returns
        -------
        scaled_disp : np.ndarray
            Scaled displacement array for watershed segmentation
        disp_mask : np.ndarray
            Mask of invalid displacement values
        """
        masked_disp = np.ma.masked_equal(input_disp, 0)
        scaled_disp = masked_disp - np.nanmin(masked_disp) + 1
        scaled_disp = np.nan_to_num(scaled_disp, nan=0)
        disp_mask = np.ma.masked_invalid(input_disp).mask

        return scaled_disp, disp_mask

    def _watershed_segmentation(
        self,
        scaled_disp: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform watershed segmentation to identify coherent regions.

        Parameters
        ----------
        scaled_disp : np.ndarray
            Scaled displacement array
        mask : np.ndarray
            Binary mask indicating valid data regions

        Returns
        -------
        labeled_regions : np.ndarray
            Labeled regions from watershed
        valid_labels : np.ndarray
            Valid region labels after area filtering
        """
        # Apply Sobel filter to create elevation map for watershed
        elevation_map = filters.sobel(mask)

        # Generate markers for watershed segmentation
        markers = np.zeros_like(scaled_disp, dtype=int)
        markers[scaled_disp < 1] = 1  # Marker for background
        markers[scaled_disp > 1] = 2  # Marker for objects

        # Perform watershed segmentation
        segmentation_result = segmentation.watershed(elevation_map, markers)
        segmentation_result = ndi.binary_fill_holes(segmentation_result - 1)
        labeled_regions, _ = ndi.label(segmentation_result)

        # Extract region properties and filter by area
        regions = measure.regionprops(labeled_regions)
        region_areas = np.array([r.area for r in regions])
        region_labels = np.array([r.label for r in regions])
        valid_labels = region_labels[region_areas > self.min_region_area]

        logger.info(f"Identified {len(valid_labels)} regions in watershed segmentation")

        return labeled_regions, valid_labels

    def _compute_regional_medians(
        self,
        input_disp: np.ndarray,
        labeled_regions: np.ndarray,
        valid_labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute median displacement for each region.

        Parameters
        ----------
        input_disp : np.ndarray
            Input displacement array
        labeled_regions : np.ndarray
            Labeled regions
        valid_labels : np.ndarray
            Valid region labels

        Returns
        -------
        np.ndarray
            Median displacement values for each region
        """
        medians = ndi.labeled_comprehension(
            input_disp,
            labeled_regions,
            index=valid_labels,
            func=np.nanmedian,
            out_dtype=np.float32,
            default=0
        )
        return medians

    def _compute_unwrap_cycles(
        self,
        medians: np.ndarray,
        ref_label: int = 0
    ) -> np.ndarray:
        """
        Compute the number of wavelength cycles to correct for each region.

        This method calculates the offset of each region relative to the
        reference region and determines the nearest integer number of
        wavelength cycles that should be removed.

        Parameters
        ----------
        medians : np.ndarray
            Median displacement values for each region
        ref_label : int, optional
            Index of reference region, by default 0

        Returns
        -------
        np.ndarray
            Number of wavelength cycles to subtract from each region
        """
        ref_median = medians[ref_label]
        # Calculate offset relative to reference
        offsets = medians - ref_median

        # Compute the number of cycles (round to nearest integer)
        # Handle NaN values by replacing them with 0
        with np.errstate(invalid='ignore'):
            cycles = np.round(offsets / self.wavelength)
        cycles = np.nan_to_num(cycles, nan=0.0).astype(np.int32)

        logger.info(f"Detected unwrap cycles per region: {cycles}")
        logger.debug(f"Wavelength corrections (in units): {cycles * self.wavelength}")

        return cycles

    def _apply_corrections(
        self,
        input_disp: np.ndarray,
        labeled_regions: np.ndarray,
        valid_labels: np.ndarray,
        unwrap_cycles: np.ndarray,
        disp_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply unwrap corrections by removing wavelength cycle offsets.

        Parameters
        ----------
        input_disp : np.ndarray
            Input displacement field
        labeled_regions : np.ndarray
            Labeled regions
        valid_labels : np.ndarray
            Valid region labels
        unwrap_cycles : np.ndarray
            Number of wavelength cycles to remove from each region
        disp_mask : np.ndarray
            Mask of invalid displacement values

        Returns
        -------
        np.ndarray
            Corrected displacement field
        """
        disp_updated = np.empty_like(input_disp, dtype=np.float32)
        _update_disp_wavelength(
            input_disp,
            labeled_regions,
            valid_labels,
            unwrap_cycles,
            self.wavelength,
            disp_updated
        )
        disp_updated = np.ma.masked_array(disp_updated, mask=disp_mask)
        return disp_updated

    def correct(
        self,
        input_disp: np.ndarray,
        mask: np.ndarray,
        ref_region: int = 0
    ) -> np.ndarray:
        """
        Correct unwrapping errors by removing wavelength cycle offsets.

        This method uses watershed segmentation to identify coherent regions
        based on the displacement field and corrects unwrap errors by detecting
        and removing integer multiples of the wavelength from each region.

        Parameters
        ----------
        input_disp : np.ndarray
            Input displacement field (unwrapped interferogram)
        mask : np.ndarray
            Binary mask indicating valid data regions
        ref_region : int, optional
            Index of reference region for correction, by default 0

        Returns
        -------
        np.ndarray
            Corrected displacement field as a masked array

        Notes
        -----
        The correction workflow:
        1. Prepare displacement data for segmentation
        2. Apply watershed segmentation to identify regions
        3. Calculate median displacement for each region
        4. Compute unwrap cycle offsets (n * wavelength)
        5. Remove cycle offsets from each region
        """
        # Prepare displacement data
        scaled_disp, disp_mask = self._prepare_displacement(input_disp)

        # Perform watershed segmentation
        labeled_regions, valid_labels = self._watershed_segmentation(
            scaled_disp,
            mask
        )

        # Compute regional medians
        medians = self._compute_regional_medians(
            input_disp,
            labeled_regions,
            valid_labels
        )

        # Compute unwrap cycles for each region
        unwrap_cycles = self._compute_unwrap_cycles(medians, ref_label=ref_region)

        # Store results for inspection
        self.labeled_regions_ = labeled_regions
        self.valid_labels_ = valid_labels
        self.medians_ = medians
        self.unwrap_cycles_ = unwrap_cycles
        self.n_regions_ = len(valid_labels)

        # Apply corrections
        corrected_disp = self._apply_corrections(
            input_disp,
            labeled_regions,
            valid_labels,
            unwrap_cycles,
            disp_mask
        )

        return corrected_disp

    def get_region_info(self) -> Optional[dict]:
        """
        Get information about identified regions.

        Returns
        -------
        dict or None
            Dictionary containing region information, or None if correction
            hasn't been run yet. Includes:
            - n_regions: Number of regions
            - valid_labels: Region labels
            - medians: Median displacement per region
            - unwrap_cycles: Number of wavelength cycles corrected per region
            - labeled_regions: Labeled region array
        """
        if self.n_regions_ is None:
            return None

        return {
            'n_regions': self.n_regions_,
            'valid_labels': self.valid_labels_,
            'medians': self.medians_,
            'unwrap_cycles': self.unwrap_cycles_,
            'labeled_regions': self.labeled_regions_,
            'wavelength': self.wavelength
        }

    def save_geotiff(
        self,
        corrected_disp: np.ndarray,
        output_path: str | Path,
        reference_file: Optional[str | Path] = None,
        transform: Optional[Affine] = None,
        crs: Optional[str] = None,
        nodata: Optional[float] = None
    ) -> None:
        """
        Save corrected displacement to a GeoTIFF file.

        Parameters
        ----------
        corrected_disp : np.ndarray
            Corrected displacement array to save
        output_path : str or Path
            Output path for the GeoTIFF file
        reference_file : str or Path, optional
            Reference raster file to copy georeferencing from
        transform : Affine, optional
            Affine transform for georeferencing (if reference_file not provided)
        crs : str, optional
            Coordinate reference system (if reference_file not provided)
        nodata : float, optional
            NoData value, by default np.nan

        Raises
        ------
        ValueError
            If neither reference_file nor (transform and crs) are provided
        ImportError
            If rasterio is not installed

        Examples
        --------
        Save with reference file for georeferencing::

            corrector = UnwrapCorrector()
            corrected = corrector.correct(disp, mask)
            corrector.save_geotiff(corrected, 'corrected.tif', reference_file='input.tif')

        Or specify georeferencing manually::

            corrector.save_geotiff(corrected, 'corrected.tif', transform=transform, crs='EPSG:4326')
        """
        if not HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for GeoTIFF output. "
                "Install it with: pip install rasterio"
            )

        output_path = Path(output_path)

        # Get georeferencing information
        if reference_file is not None:
            # Copy from reference file
            with rio.open(reference_file) as src:
                transform = src.transform
                crs = src.crs
                if nodata is None:
                    nodata = src.nodata
        elif transform is None or crs is None:
            raise ValueError(
                "Either reference_file or both transform and crs must be provided"
            )

        # Set default nodata if not specified
        if nodata is None:
            nodata = np.nan

        # Handle masked arrays
        if np.ma.isMaskedArray(corrected_disp):
            data = corrected_disp.filled(nodata)
        else:
            data = corrected_disp

        # Write GeoTIFF
        with rio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)

        logger.info(f"Saved corrected displacement to: {output_path}")


def read_data(file: str | Path) -> np.ndarray:
    """
    Read raster data from a file.

    Parameters
    ----------
    file : str or Path
        Path to the raster file

    Returns
    -------
    np.ndarray
        2D array containing the raster data
    """
    with rio.open(file) as temp_src:
        data = temp_src.read(1)
    return data


def read_netcdf(
    file: str | Path,
    disp_variable: str = 'displacement',
    mask_variable: str = 'water_mask'
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Read displacement and mask data from a NetCDF file with georeferencing.

    Parameters
    ----------
    file : str or Path
        Path to the NetCDF file
    disp_variable : str, optional
        Displacement variable name, by default 'displacement'
    mask_variable : str, optional
        Mask variable name, by default 'water_mask'

    Returns
    -------
    data : np.ndarray
        2D displacement array
    mask : np.ndarray
        2D mask array
    geo_info : dict
        Dictionary containing georeferencing information:
        - 'transform': rasterio Affine transform
        - 'crs': Coordinate reference system
        - 'nodata': NoData value if available

    Raises
    ------
    ValueError
        If the file cannot be read or lacks required variables/georeferencing
    ImportError
        If rasterio is required for georeferencing but not installed
    """
    import xarray as xr
    if HAS_RASTERIO:
        from rasterio.transform import from_bounds
    else:
        from_bounds = None

    ds = xr.open_dataset(file)

    # Read displacement variable
    if disp_variable not in ds:
        raise ValueError(f"Displacement variable '{disp_variable}' not found in {file}")

    data_array = ds[disp_variable]
    data = data_array.values

    # Squeeze out any singleton dimensions
    data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D displacement data, got {data.ndim}D")

    # Read mask variable
    if mask_variable not in ds:
        raise ValueError(f"Mask variable '{mask_variable}' not found in {file}")

    mask_array = ds[mask_variable]
    mask = mask_array.values

    # Squeeze out any singleton dimensions
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask data, got {mask.ndim}D")

    # Check that mask and data have the same shape
    if data.shape != mask.shape:
        raise ValueError(
            f"Displacement and mask shapes do not match: "
            f"{data.shape} vs {mask.shape}"
        )

    # Extract georeferencing information
    geo_info = {}

    # Get CRS - try multiple sources
    if 'spatial_ref' in ds:
        sr = ds['spatial_ref']
        # Try different CRS attribute names
        for crs_attr in ['crs_wkt', 'spatial_ref', 'wkt']:
            if crs_attr in sr.attrs:
                geo_info['crs'] = sr.attrs[crs_attr]
                logger.debug(f"CRS from spatial_ref.{crs_attr}")
                break
        # If we found crs_wkt but want EPSG code, try to extract it
        if 'crs_wkt' in sr.attrs and 'projected_crs_name' in sr.attrs:
            logger.debug(f"Projected CRS: {sr.attrs['projected_crs_name']}")

    if 'crs' not in geo_info and 'crs' in ds.attrs:
        geo_info['crs'] = ds.attrs['crs']
        logger.debug(f"CRS from global attributes")

    if 'crs' not in geo_info:
        # Try to infer from coordinate names
        geo_info['crs'] = 'EPSG:4326'  # Default assumption
        logger.warning("No CRS found in NetCDF, assuming EPSG:4326")

    # Get transform - check for GeoTransform first (GDAL-style)
    if 'spatial_ref' in ds and 'GeoTransform' in ds['spatial_ref'].attrs:
        if not HAS_RASTERIO:
            logger.warning("rasterio not available, skipping transform extraction")
        else:
            # Parse GeoTransform string: "top_left_x pixel_width 0 top_left_y 0 -pixel_height"
            gt_str = ds['spatial_ref'].attrs['GeoTransform']
            gt = [float(x) for x in gt_str.split()]
            transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
            geo_info['transform'] = transform
            logger.debug(f"Created transform from GeoTransform: {transform}")
    elif 'x' in ds.coords and 'y' in ds.coords:
        if not HAS_RASTERIO:
            logger.warning("rasterio not available, skipping transform extraction")
        else:
            x = ds.coords['x'].values
            y = ds.coords['y'].values

            # Calculate pixel size
            x_res = abs(x[1] - x[0]) if len(x) > 1 else 1.0
            y_res = abs(y[1] - y[0]) if len(y) > 1 else 1.0

            # Get bounds
            x_min = float(x.min()) - x_res / 2
            x_max = float(x.max()) + x_res / 2
            y_min = float(y.min()) - y_res / 2
            y_max = float(y.max()) + y_res / 2

            # Create transform
            transform = from_bounds(x_min, y_min, x_max, y_max, len(x), len(y))
            geo_info['transform'] = transform
            logger.debug(f"Created transform from x/y coordinates: {transform}")
    elif 'lon' in ds.coords and 'lat' in ds.coords:
        if not HAS_RASTERIO:
            logger.warning("rasterio not available, skipping transform extraction")
        else:
            lon = ds.coords['lon'].values
            lat = ds.coords['lat'].values

            lon_res = abs(lon[1] - lon[0]) if lon.ndim == 1 and len(lon) > 1 else 1.0
            lat_res = abs(lat[1] - lat[0]) if lat.ndim == 1 and len(lat) > 1 else 1.0

            if lon.ndim == 1:
                lon_min = float(lon.min()) - lon_res / 2
                lon_max = float(lon.max()) + lon_res / 2
                lat_min = float(lat.min()) - lat_res / 2
                lat_max = float(lat.max()) + lat_res / 2

                transform = from_bounds(lon_min, lat_min, lon_max, lat_max, len(lon), len(lat))
                geo_info['transform'] = transform
                logger.debug(f"Created transform from lon/lat coordinates: {transform}")
    else:
        available_coords = list(ds.coords.keys())
        raise ValueError(
            f"Could not find coordinate information (x/y or lon/lat) in NetCDF file. "
            f"Available coordinates: {available_coords}"
        )

    # Get nodata value
    if hasattr(data_array, '_FillValue'):
        geo_info['nodata'] = float(data_array._FillValue)
    elif hasattr(data_array, 'missing_value'):
        geo_info['nodata'] = float(data_array.missing_value)
    else:
        geo_info['nodata'] = np.nan

    ds.close()

    return data, mask, geo_info


def correct_region_offset(
    input_disp: Union[np.ndarray, str, Path],
    mask: Optional[np.ndarray] = None,
    wavelength: float = 0.0555,
    min_region_area: int = 20,
    output_file: Optional[str | Path] = None
) -> np.ndarray:
    """
    Correct unwrapping errors by removing wavelength cycle offsets.

    This is a convenience function that wraps the UnwrapCorrector class
    for simple use cases. It detects and removes integer multiples of the
    wavelength from different regions based on the displacement field.

    Parameters
    ----------
    input_disp : np.ndarray or str or Path
        Input displacement field (unwrapped interferogram) as array,
        or path to NetCDF file (.nc) to read. If NetCDF file, expects
        'displacement' and 'water_mask' variables.
    mask : np.ndarray, optional
        Binary mask indicating valid data regions. If None and input_disp
        is a NetCDF file, mask will be read from 'water_mask' variable.
        Required if input_disp is an array.
    wavelength : float, optional
        The wavelength value for unwrap correction, by default 0.0555m for Sentinel-1.
        Common values:
        - Sentinel-1 (C-band): 0.0555 m
        - ALOS-2 (L-band): 0.236 m
        - Phase data (radians): 2π (6.283)
    min_region_area : int, optional
        Minimum area (in pixels) for valid regions, by default 20
    output_file : str or Path, optional
        If provided, save the corrected displacement to this GeoTIFF file.
        Georeferencing is automatically extracted from input NetCDF file.

    Returns
    -------
    np.ndarray
        Corrected displacement field as a masked array

    Examples
    --------
    From NetCDF file (mask auto-loaded from 'water_mask' variable)::

        corrected = correct_region_offset(
            'displacement.nc',
            output_file='corrected.tif'
        )

    From array with explicit mask::

        corrected = correct_region_offset(disp, mask)

    From NetCDF with custom output::

        corrected = correct_region_offset(
            'displacement.nc',
            output_file='corrected.tif',
            wavelength=0.0555
        )

    For phase data (radians)::

        corrected = correct_region_offset(phase, mask, wavelength=2*np.pi)

    For ALOS-2 displacement (meters)::

        corrected = correct_region_offset(disp, mask, wavelength=0.236)

    Notes
    -----
    NetCDF files must contain:

    - 'displacement' variable: unwrapped displacement data
    - 'water_mask' variable: binary mask (1=valid, 0=invalid)
    - Coordinate information (x/y or lon/lat)

    For more control and access to intermediate results, use the
    UnwrapCorrector class directly::

        corrector = UnwrapCorrector(wavelength=0.0555, min_region_area=20)
        corrected = corrector.correct(disp, mask)
        corrector.save_geotiff(corrected, 'corrected.tif', reference_file='input.tif')
        region_info = corrector.get_region_info()
        print(f"Corrected {region_info['n_regions']} regions")
        print(f"Unwrap cycles: {region_info['unwrap_cycles']}")
    """
    # Check if input is a file path
    geo_info = None
    if isinstance(input_disp, (str, Path)):
        input_path = Path(input_disp)
        if input_path.suffix.lower() == '.nc':
            logger.info(f"Reading NetCDF file: {input_path}")
            disp_data, mask_data, geo_info = read_netcdf(input_path)
            # Use mask from file if not provided
            if mask is None:
                mask = mask_data
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .nc for NetCDF files.")
    else:
        disp_data = input_disp
        if mask is None:
            raise ValueError("mask parameter is required when input_disp is an array")

    # Apply mask to displacement data
    logger.info("Applying mask to displacement data...")

    # Create masked array - set invalid values to NaN
    disp_masked = np.where(mask.astype(bool), disp_data, np.nan)

    n_valid = np.sum(~np.isnan(disp_masked))
    n_total = disp_masked.size
    logger.info(f"Valid pixels: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    # Run correction
    corrector = UnwrapCorrector(
        min_region_area=min_region_area,
        wavelength=wavelength
    )
    corrected = corrector.correct(disp_masked, mask)

    # Save to GeoTIFF if output file is specified
    if output_file is not None:
        if geo_info is None:
            raise ValueError(
                "Cannot save to GeoTIFF without georeferencing information. "
                "Provide input_disp as a NetCDF file path to automatically extract georeferencing."
            )

        # Check that transform and crs were successfully extracted
        if 'transform' not in geo_info or geo_info['transform'] is None:
            raise ValueError(
                "Failed to extract transform from NetCDF file. "
                "Ensure the file has valid coordinate information (x/y or lon/lat)."
            )
        if 'crs' not in geo_info or geo_info['crs'] is None:
            raise ValueError(
                "Failed to extract CRS from NetCDF file. "
                "Ensure the file has valid coordinate reference system information."
            )

        corrector.save_geotiff(
            corrected,
            output_file,
            transform=geo_info['transform'],
            crs=geo_info['crs'],
            nodata=geo_info.get('nodata', np.nan)
        )

    return corrected
