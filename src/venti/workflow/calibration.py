"""Calibration workflow for InSAR displacement products.

This module provides a high-level CalibrationWorkflow class that orchestrates
the entire calibration process using dataclasses and object composition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from .config import WorkflowConfig

if TYPE_CHECKING:
    from ..gnss.reference import GNSSReference
    from ..io.read import RasterReader
    from ..io.write import RasterWriter
    from ..spatial.processor import SpatialProcessor

logger = logging.getLogger(__name__)

# Half-wavelength for Sentinel-1 C-band in mm: λ/2 = 0.0555/2 * 1000
_WAVELENGTH_MM: float = 0.0555 / 2 * 1000


@dataclass
class CalibrationState:
    """State tracking for calibration workflow.

    Attributes
    ----------
    n_files_total : int
        Total number of files to process
    n_files_processed : int
        Number of files processed
    n_files_failed : int
        Number of files that failed
    output_files : list[Path]
        List of output file paths

    """

    n_files_total: int = 0
    n_files_processed: int = 0
    n_files_failed: int = 0
    output_files: list[Path] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if self.n_files_total == 0:
            return 0.0
        return 100.0 * self.n_files_processed / self.n_files_total

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.n_files_processed == 0:
            return 0.0
        return (
            100.0
            * (self.n_files_processed - self.n_files_failed)
            / self.n_files_processed
        )


@dataclass
class CalibrationWorkflow:
    """High-level workflow manager for InSAR calibration.

    This class orchestrates the entire calibration process using GNSS reference data,
    combining GNSS data management, spatial processing, and I/O operations.

    Attributes
    ----------
    config : WorkflowConfig
        Workflow configuration
    gnss_manager : GNSSReference
        GNSS reference data interface
    io_reader : RasterReader
        I/O reader for loading raster data
    io_writer : RasterWriter
        I/O writer for saving raster data
    spatial_processor : SpatialProcessor
        Spatial processor
    state : CalibrationState
        Workflow state tracking

    """

    config: WorkflowConfig
    gnss_manager: GNSSReference | None = field(default=None, init=False)
    io_reader: RasterReader | None = field(default=None, init=False)
    io_writer: RasterWriter | None = field(default=None, init=False)
    spatial_processor: SpatialProcessor | None = field(default=None, init=False)
    state: CalibrationState = field(default_factory=CalibrationState, init=False)

    def __post_init__(self):
        """Initialize workflow components."""
        from ..io.read import RasterReader
        from ..io.write import RasterWriter
        from ..spatial.processor import SpatialProcessor

        ## TODO: the imports above are based on the current implementation
        # transferred from calibrate_timeseries.py. They might change if we use
        # xarray spatial module and geepers for GNSS
        # same for functions imported from utils, mainly downsample and upsample

        # Create output directory
        self.config.run_config.product_path_group.product_path.mkdir(
            parents=True, exist_ok=True
        )

        # Initialize components
        self.io_reader = RasterReader()
        self.io_writer = RasterWriter()
        self.spatial_processor = SpatialProcessor()

        # Store worker settings for use in processing methods
        self.gpu_enabled = self.config.worker_settings.gpu_enabled
        self.block_shape = self.config.worker_settings.block_shape
        self.threads_per_worker = self.config.worker_settings.threads_per_worker

        # Match correction files to displacement files if provided
        self.matched_files = None

        logger.info("Calibration workflow initialized")

    def setup_gnss(self) -> int:
        """Set up GNSS reference data and download stations.

        Returns
        -------
        int
            Number of GNSS stations downloaded

        """
        from ..gnss.reference import GNSSReference

        assert self.io_reader is not None, "io_reader not initialized"

        # Get bounds from first displacement file
        disp_files = sorted(self.config.input_options.input_files.glob("*.nc"))
        if not disp_files:
            msg = f"No NetCDF files in {self.config.input_options.input_files}"
            raise FileNotFoundError(msg)

        bounds = self.io_reader.get_bounds(disp_files[0], as_latlon=False)

        # Get UTM EPSG
        from pyproj import CRS

        netcdf_data = self.io_reader.read_netcdf(disp_files[0])
        utm_epsg = CRS.from_user_input(netcdf_data.crs).to_epsg()

        # Initialize GNSS reference
        gnss_dir = self.config.run_config.product_path_group.product_path / "GNSS"
        self.gnss_manager = GNSSReference(
            bounds=bounds,
            output_dir=gnss_dir,
            reference_frame=self.config.grid_settings.reference_frame,
            utm_epsg=utm_epsg,
        )

        # Download stations
        n_stations = self.gnss_manager.download_stations()

        if n_stations == 0:
            msg = "No GNSS stations found in area"
            raise ValueError(msg)

        logger.info(f"GNSS setup complete: {n_stations} stations")
        return n_stations

    def load_los_and_mask(self) -> tuple:
        """Load LOS vectors and mask.

        Returns
        -------
        tuple
            (los_east, los_north, los_up, mask)

        """
        assert self.io_reader is not None, "io_reader not initialized"

        logger.info("Loading LOS unit vectors and mask...")

        # Load LOS
        los_data = self.io_reader.read_geotiff(self.config.input_options.los_file)
        if los_data.data.ndim == 3:
            los_east = los_data.data[0]
            los_north = los_data.data[1]
            los_up = los_data.data[2]
        else:
            msg = "LOS file must be 3-band (east, north, up)"
            raise ValueError(msg)

        # Load mask
        mask_data = self.io_reader.read_geotiff(self.config.input_options.water_mask)
        mask = mask_data.data.astype(bool)

        logger.info("LOS and mask loaded")
        return los_east, los_north, los_up, mask

    def find_reference_point(self, mask: np.ndarray) -> tuple[int, int]:
        """Find or use configured reference point.

        Parameters
        ----------
        mask : np.ndarray
            Boolean valid-pixel mask (True = valid). Masked pixels are excluded
            from auto-selection by zeroing their coherence before scoring.

        Returns
        -------
        tuple
            (row, col) reference point

        """
        if self.config.input_options.reference_point is not None:
            logger.info(
                "Using configured reference point:"
                f" {self.config.input_options.reference_point}"
            )
            return self.config.input_options.reference_point

        import tempfile

        import rasterio

        from .utils import compute_average_temporal_coherence

        disp_files = sorted(self.config.input_options.input_files.glob("*.nc"))
        coherence_file = compute_average_temporal_coherence(
            disp_files,
            self.config.run_config.product_path_group.product_path,
            variable="temporal_coherence",
        )

        # Zero out invalid pixels so they cannot be selected as reference
        with rasterio.open(coherence_file) as src:
            profile = src.profile.copy()
            coherence = src.read(1)

        valid_mask = mask.squeeze().astype(bool)
        assert coherence.shape == valid_mask.shape, (
            f"Coherence shape {coherence.shape} does not match "
            f"mask shape {valid_mask.shape}"
        )
        coherence_masked = np.where(valid_mask, coherence, 0.0).astype(profile["dtype"])

        from opera_utils.disp import rebase_reference

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            masked_path = Path(tmp.name)

        try:
            with rasterio.open(masked_path, "w", **profile) as dst:
                dst.write(coherence_masked, 1)
            ref_point = rebase_reference.find_reference_point(masked_path)
        finally:
            masked_path.unlink(missing_ok=True)

        logger.info(f"Auto-selected reference point: {ref_point}")
        return ref_point

    def compute_gnss_reference(
        self,
        los_east: np.ndarray,
        los_north: np.ndarray,
        los_up: np.ndarray,
        disp_file: Path,
        ref_date: float | None = None,
        sec_date: float | None = None,
    ) -> np.ndarray:
        """Compute GNSS reference in LOS.

        Parameters
        ----------
        los_east : np.ndarray
            LOS east component
        los_north : np.ndarray
            LOS north component
        los_up : np.ndarray
            LOS up component
        disp_file : Path
            Displacement file for dates
        ref_date : float, optional
            Reference date
        sec_date : float, optional
            Secondary date

        Returns
        -------
        np.ndarray
            GNSS LOS displacement/velocity

        """
        assert self.gnss_manager is not None, "gnss_manager not initialized"

        output_dir = self.config.run_config.product_path_group.product_path
        recompute = self.config.grid_settings.recompute_gnss

        if self.config.grid_settings.grid_type == "constant":
            if not hasattr(self, "_gnss_velocity"):
                cache_file = output_dir / "gnss_los_velocity.npy"
                if cache_file.exists() and not recompute:
                    logger.info(f"Loading cached GNSS LOS velocity from {cache_file}")
                    self._gnss_velocity = np.load(cache_file)
                else:
                    logger.info(
                        "Computing GNSS LOS velocity from"
                        f" {self.config.grid_settings.starting_year}"
                    )
                    self._gnss_velocity = self.gnss_manager.compute_velocity_los(
                        los_east=los_east,
                        los_north=los_north,
                        los_up=los_up,
                        netcdf_file=disp_file,
                        start_year=self.config.grid_settings.starting_year,
                        method="rbf",
                    )
                    np.save(cache_file, self._gnss_velocity)
                    logger.info(f"Saved GNSS LOS velocity to {cache_file}")

            if ref_date is None or sec_date is None:
                return self._gnss_velocity
            # sec_date > ref_date (OPERA convention), so (sec_date - ref_date) > 0.
            return self._gnss_velocity * (sec_date - ref_date)

        else:
            if ref_date is None or sec_date is None:
                msg = "ref_date and sec_date required for variable grid type"
                raise ValueError(msg)

            cache_file = output_dir / f"gnss_los_disp_{ref_date:.4f}_{sec_date:.4f}.npy"
            if cache_file.exists() and not recompute:
                logger.info(f"Loading cached GNSS LOS displacement from {cache_file}")
                return np.load(cache_file)

            result = self.gnss_manager.compute_displacement_los(
                ref_date=ref_date,
                sec_date=sec_date,
                los_east=los_east,
                los_north=los_north,
                los_up=los_up,
                netcdf_file=disp_file,
                method="rbf",
            )
            np.save(cache_file, result)
            logger.info(f"Saved GNSS LOS displacement to {cache_file}")
            return result

    def process_displacement_file(
        self,
        disp_file: Path,
        los_east: np.ndarray,
        los_north: np.ndarray,
        los_up: np.ndarray,
        mask: np.ndarray,
        ref_point: tuple[int, int],
        window_size_x: int,
        window_size_y: int,
        tropo_file: Path | None = None,
        event_mask_file: Path | None = None,
    ) -> Path | None:
        """Process a single displacement file.

        Parameters
        ----------
        disp_file : Path
            Displacement file path
        los_east : np.ndarray
            LOS east component
        los_north : np.ndarray
            LOS north component
        los_up : np.ndarray
            LOS up component
        mask : np.ndarray
            Valid pixel mask
        ref_point : tuple
            Reference point (row, col)
        window_size_x : int
            Window width
        window_size_y : int
            Window height
        tropo_file : Path, optional
            Tropospheric correction file path
        event_mask_file : Path, optional
            Per-epoch event mask GeoTIFF (1=valid, 0=event region).  When
            provided, the event region is filled with nearest valid neighbors
            before calibration surface estimation.  The calibration surface is
            then removed from the full (unmasked) displacement.

        Returns
        -------
        Path or None
            Output file path if successful

        """
        from ..unwrap import correct_region_offset
        from .utils import downsample_array, get_file_dates, upsample_array

        assert self.io_reader is not None, "io_reader not initialized"
        assert self.io_writer is not None, "io_writer not initialized"
        assert self.spatial_processor is not None, "spatial_processor not initialized"

        try:
            # Extract dates
            ref_date, sec_date = get_file_dates(disp_file)
        except Exception:
            logger.warning(f"Could not extract dates from {disp_file.name}")
            return None

        logger.debug(f"Processing {disp_file.name}")

        # Read displacement
        netcdf_data = self.io_reader.read_netcdf(disp_file, variable="displacement")
        disp = netcdf_data.data * 1000  # Convert to mm
        refy, refx = ref_point
        disp -= disp[refy, refx]

        # Apply tropospheric correction if available
        if tropo_file is not None:
            tropo_data = self.io_reader.read_geotiff(tropo_file)
            tropo_corr = tropo_data.data * 1000  # Convert to mm
            tropo_corr -= tropo_corr[refy, refx]
            disp -= tropo_corr

        # Get GNSS LOS
        gnss_los = self.compute_gnss_reference(
            los_east, los_north, los_up, disp_file, ref_date, sec_date
        )
        gnss_los -= gnss_los[refy, refx]

        # Apply mask
        disp = np.where(mask & ~np.isnan(disp), disp, np.nan)

        # Load event mask and build displacement array for calibration fitting.
        event_mask: np.ndarray | None = None
        if event_mask_file is not None:
            event_mask_data = self.io_reader.read_geotiff(event_mask_file)
            event_mask = event_mask_data.data.astype(bool)

            buffer_px = (
                self.config.algorithm_parameters.calibration_options.event_mask_buffer_pixels
            )
            if buffer_px > 0:
                from scipy.ndimage import binary_dilation

                # Dilate the event region (False pixels) outward by buffer_px pixels
                event_mask = ~binary_dilation(~event_mask, iterations=buffer_px)
                logger.info(
                    f"Event mask boundary buffered by {buffer_px} px: "
                    f"{int((~event_mask).sum()):,} total event-region pixels"
                )

            n_event_pixels = int((~event_mask).sum())
            logger.debug(
                f"Event mask loaded from {event_mask_file.name}: "
                f"{n_event_pixels:,} event-region pixels will be filled for calibration"
            )

        # Correct unwrap errors (optional)
        apply_unwrap_correction = (
            self.config.algorithm_parameters.calibration_options.unwrap_error_correction
        )
        if apply_unwrap_correction:
            logger.debug("Correcting unwrap errors...")
            disp = correct_region_offset(
                input_disp=disp, mask=mask, wavelength=_WAVELENGTH_MM
            )

        if isinstance(disp, np.ma.MaskedArray):
            disp = disp.filled(np.nan)
        if isinstance(gnss_los, np.ma.MaskedArray):
            gnss_los = gnss_los.filled(np.nan)

        # Build displacement array for calibration surface estimation.
        # If an event mask is provided, fill the event region with nearest pixels
        if event_mask is not None:
            from ..spatial.interpolation import fill_masked_region

            disp_for_cal = fill_masked_region(disp, event_mask)
        else:
            disp_for_cal = disp

        # Downsample if requested
        original_shape = disp.shape
        if self.config.grid_settings.downsample_factor > 1:
            logger.info(
                f"Downsampling by factor {self.config.grid_settings.downsample_factor} "
                f"using method '{self.config.grid_settings.downsample_method}'"
            )

            # Load weights if weighted downsampling is requested
            weights = None
            if self.config.grid_settings.downsample_weighted:
                try:
                    # Try to load temporal coherence as weights
                    coh_data = self.io_reader.read_netcdf(
                        disp_file, variable="temporal_coherence"
                    )
                    weights = coh_data.data
                    logger.info("Downsampling weighted by temporal coherence")
                except Exception as e:
                    logger.warning(
                        f"Could not load weights for downsampling: {e}. "
                        "Using unweighted downsampling."
                    )

            disp_for_cal_ds = downsample_array(
                disp_for_cal,
                self.config.grid_settings.downsample_factor,
                method=self.config.grid_settings.downsample_method,
                weights=weights,
            )

            # Downsample GNSS LOS
            gnss_los_ds = downsample_array(
                gnss_los,
                self.config.grid_settings.downsample_factor,
                method=self.config.grid_settings.downsample_method,
                weights=weights,
            )

            win_x_ds = max(
                1, window_size_x // self.config.grid_settings.downsample_factor
            )
            win_y_ds = max(
                1, window_size_y // self.config.grid_settings.downsample_factor
            )
        else:
            disp_for_cal_ds = disp_for_cal
            gnss_los_ds = gnss_los
            win_x_ds = window_size_x
            win_y_ds = window_size_y

        # Fit calibration surface using event-filled displacement so that
        # transient deformation in the event region does not bias the fit
        logger.debug("Fitting calibration surface...")
        # Overlap of 50 % ensures Hann-tapered windows sum to near-uniform weight.
        overlap_x = win_x_ds // 2
        overlap_y = win_y_ds // 2
        cal_opts = self.config.algorithm_parameters.calibration_options
        smoothing_method = cal_opts.calibration_surface_smoothing_method
        cfg_sigma = cal_opts.calibration_surface_smoothing_sigma
        if cfg_sigma is None:
            # Default: 1/8 of the smaller window dimension suppresses seams
            # without over-smoothing.
            smoothing_sigma: float | None = min(win_x_ds, win_y_ds) / 8
        elif cfg_sigma == 0:
            smoothing_sigma = None
        else:
            smoothing_sigma = cfg_sigma
        calibration_surface = self.spatial_processor.fit_windowed_surface(
            insar_data=disp_for_cal_ds,
            gnss_los=gnss_los_ds,
            window_size_x=win_x_ds,
            window_size_y=win_y_ds,
            window_overlap_x=overlap_x,
            window_overlap_y=overlap_y,
            poly_order=1.5,
            n_jobs=-1,
            smoothing_sigma=smoothing_sigma,
            smoothing_method=smoothing_method,
            sg_window_length=cal_opts.savitzky_golay.window_length,
            sg_polyorder=cal_opts.savitzky_golay.polyorder,
        )

        # Upsample calibration surface if needed
        if self.config.grid_settings.downsample_factor > 1:
            calibration_surface_full = upsample_array(
                calibration_surface, original_shape
            )
        else:
            calibration_surface_full = calibration_surface

        # Convert back to meters
        calibration_surface_full = calibration_surface_full / 1000.0

        # Build output filename
        grid_type = self.config.grid_settings.grid_type
        ref_frame = self.config.grid_settings.reference_frame.lower()
        suffix = f"_calibration_surface_{grid_type}_{ref_frame}"
        if self.config.grid_settings.downsample_factor > 1:
            suffix += f"_downsample{self.config.grid_settings.downsample_factor}"
        if tropo_file is not None:
            suffix += "_tropo"
        if not apply_unwrap_correction:
            suffix += "_nowrap"

        output_file = (
            self.config.run_config.product_path_group.product_path
            / f"{disp_file.stem}{suffix}.tif"
        )

        # Save
        description = (
            f"Calibration surface ({self.config.grid_settings.grid_type} GNSS model)"
        )
        if tropo_file is not None:
            description += " with tropospheric correction"

        self.io_writer.write_geotiff(
            calibration_surface_full,
            output_file,
            reference_file=disp_file,
            nodata=np.nan,
            descriptions=[description],
        )

        logger.debug(f"Saved: {output_file.name}")
        return output_file

    def _find_event_mask_file(self, disp_file: Path) -> Path | None:
        """Find the per-epoch event mask file matching a displacement file.

        Looks for a GeoTIFF in ``config.input_options.event_mask_dir`` whose
        filename starts with the displacement file stem, following the naming
        convention produced by ``generate_event_mask.py``:
        ``<disp_stem>_<geojson_stem>_mask.tif``.

        Parameters
        ----------
        disp_file : Path
            Displacement NetCDF file for which to find a matching event mask.

        Returns
        -------
        Path or None
            Path to the matching event mask GeoTIFF, or ``None`` if no
            ``event_mask_dir`` is configured or no match is found.

        """
        event_mask_dir = self.config.input_options.event_mask_dir
        if event_mask_dir is None:
            return None

        matches = sorted(event_mask_dir.glob(f"{disp_file.stem}_*mask.tif"))
        if not matches:
            logger.debug(f"No event mask found for {disp_file.name}")
            return None

        if len(matches) > 1:
            logger.warning(
                f"Multiple event masks found for {disp_file.name}; "
                f"using {matches[0].name}"
            )
        return matches[0]

    def run_single(
        self,
        disp_file: Path,
        tropo_file: Path | None = None,
    ) -> CalibrationState:
        """Run the calibration workflow on a single displacement file.

        Performs the same setup as `run` (GNSS download, LOS/mask loading,
        reference point selection) but processes only the one specified file.

        Parameters
        ----------
        disp_file : Path
            Path to the NetCDF displacement file to calibrate.
        tropo_file : Path, optional
            Path to a tropospheric correction GeoTIFF for this epoch.

        Returns
        -------
        CalibrationState
            Workflow state with ``n_files_total = 1`` and, on success,
            one entry in ``output_files``.

        """
        from .utils import parse_window_size_meters

        assert self.io_reader is not None, "io_reader not initialized"

        logger.info("=" * 60)
        logger.info("Starting Venti Single-File Calibration")
        logger.info("=" * 60)
        logger.info(f"Input file : {disp_file}")
        if tropo_file is not None:
            logger.info(f"Tropo file : {tropo_file}")

        self.setup_gnss()
        los_east, los_north, los_up, mask = self.load_los_and_mask()
        ref_point = self.find_reference_point(mask)

        window_size_pixels = parse_window_size_meters(
            self.config.grid_settings.window_size_meters,
            self.config.grid_settings.posting_meters,
        )

        self.state.n_files_total = 1

        event_mask_file = self._find_event_mask_file(disp_file)
        if event_mask_file is not None:
            logger.info(f"Event mask : {event_mask_file}")

        output_file = self.process_displacement_file(
            disp_file,
            los_east,
            los_north,
            los_up,
            mask,
            ref_point,
            window_size_pixels,
            window_size_pixels,
            tropo_file=tropo_file,
            event_mask_file=event_mask_file,
        )

        if output_file:
            self.state.output_files.append(output_file)
        else:
            self.state.n_files_failed += 1

        self.state.n_files_processed += 1

        logger.info("=" * 60)
        logger.info("Single-File Calibration Complete")
        logger.info("=" * 60)
        logger.info(f"Output: {output_file or 'FAILED'}")

        return self.state

    def run(self, max_files: int | None = None) -> CalibrationState:
        """Run the complete calibration workflow.

        Parameters
        ----------
        max_files : int or None, optional
            Processing only the first ``max_files`` displacement files.
            Set the number of displacement files to calibrate.
            Default is None (process all files).

        Returns
        -------
        CalibrationState
            Final workflow state

        """
        assert self.io_reader is not None, "io_reader not initialized"

        logger.info("=" * 60)
        logger.info("Starting Venti Calibration Workflow")
        logger.info("=" * 60)

        # Setup GNSS
        self.setup_gnss()

        # Load LOS and mask
        los_east, los_north, los_up, mask = self.load_los_and_mask()

        # Find reference point
        ref_point = self.find_reference_point(mask)

        # Get displacement files
        disp_files = sorted(self.config.input_options.input_files.glob("*.nc"))
        if max_files is not None:
            disp_files = disp_files[:max_files]
        self.state.n_files_total = len(disp_files)

        logger.info(f"Processing {self.state.n_files_total} displacement files")

        # Match correction files if provided
        from .utils import match_correction_to_displacement

        if self.config.input_options.tropo_files is not None:
            tropo_files = sorted(
                self.config.input_options.tropo_files.glob("tropo_corr*.tif")
            )
            self.matched_files = match_correction_to_displacement(
                tropo_files, disp_files
            )
            logger.info(
                f"Matched {len(self.matched_files)} tropospheric correction files"
            )
        else:
            self.matched_files = match_correction_to_displacement(None, disp_files)

        # Calculate window size
        from .utils import parse_window_size_meters

        window_size_pixels = parse_window_size_meters(
            self.config.grid_settings.window_size_meters,
            self.config.grid_settings.posting_meters,
        )

        # Process each file
        for tropo_file, disp_file in tqdm(self.matched_files, desc="Calibrating"):
            event_mask_file = self._find_event_mask_file(disp_file)
            output_file = self.process_displacement_file(
                disp_file,
                los_east,
                los_north,
                los_up,
                mask,
                ref_point,
                window_size_pixels,
                window_size_pixels,
                tropo_file=tropo_file,
                event_mask_file=event_mask_file,
            )

            if output_file:
                self.state.output_files.append(output_file)
            else:
                self.state.n_files_failed += 1

            self.state.n_files_processed += 1

        # Summary
        logger.info("=" * 60)
        logger.info("Calibration Workflow Complete!")
        logger.info("=" * 60)
        logger.info(f"Total files: {self.state.n_files_total}")
        logger.info(f"Processed: {self.state.n_files_processed}")
        logger.info(f"Failed: {self.state.n_files_failed}")
        logger.info(f"Success rate: {self.state.success_rate:.1f}%")
        logger.info(
            "Output directory:"
            f" {self.config.run_config.product_path_group.product_path}"
        )

        return self.state


def run_calibration_workflow(config: WorkflowConfig) -> CalibrationState:
    """Run calibration workflow from configuration.

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration

    Returns
    -------
    CalibrationState
        Final workflow state

    """
    workflow = CalibrationWorkflow(config=config)
    return workflow.run()
