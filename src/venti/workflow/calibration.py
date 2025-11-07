"""Object-oriented calibration workflow for InSAR displacement products.

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
        self.config.input_options.work_directory.mkdir(parents=True, exist_ok=True)

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
        netcdf_data = self.io_reader.read_netcdf(disp_files[0])
        utm_crs = netcdf_data.crs

        if hasattr(utm_crs, "to_epsg"):
            utm_epsg = utm_crs.to_epsg()
        else:
            import re

            match = re.search(r"EPSG:(\d+)", str(utm_crs))
            utm_epsg = int(match.group(1)) if match else None

        # Initialize GNSS reference
        gnss_dir = self.config.input_options.work_directory / "GNSS"
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
            Valid pixel mask

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

        # Compute average coherence for auto-selection
        from .utils import compute_average_temporal_coherence

        disp_files = sorted(self.config.input_options.input_files.glob("*.nc"))
        coherence_file = compute_average_temporal_coherence(
            disp_files,
            self.config.input_options.work_directory,
            variable="temporal_coherence",
        )

        try:
            from opera_utils.disp import rebase_reference

            ref_point = rebase_reference.find_reference_point(coherence_file)
        except Exception as e:
            logger.warning(f"Auto-selection failed: {e}, using center")
            return (mask.shape[0] // 2, mask.shape[1] // 2)
        else:
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

        if self.config.grid_settings.grid_type == "constant":
            # Use velocities
            if not hasattr(self, "_gnss_velocity"):
                logger.info(
                    "Computing GNSS velocities from"
                    f" {self.config.grid_settings.starting_year}"
                )
                gnss_velocities = self.gnss_manager.compute_velocities(
                    start_year=self.config.grid_settings.starting_year
                )
                gnss_los = gnss_velocities.project_to_los(
                    los_east, los_north, los_up, disp_file, method="rbf"
                )
                self._gnss_velocity = gnss_los

            # Scale by time span
            if ref_date and sec_date:
                time_span = ref_date - sec_date
                return self._gnss_velocity * time_span
            else:
                return self._gnss_velocity

        else:
            # Compute epoch-specific displacement
            if ref_date is None or sec_date is None:
                msg = "ref_date and sec_date required for variable grid type"
                raise ValueError(msg)

            gnss_disp = self.gnss_manager.compute_displacement(ref_date, sec_date)
            return gnss_disp.project_to_los(
                los_east, los_north, los_up, disp_file, method="rbf"
            )

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
        bounds: tuple,
        tropo_file: Path | None = None,
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
        bounds : tuple
            Geographic bounds
        tropo_file : Path, optional
            Tropospheric correction file path

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

        # Correct unwrap errors
        logger.debug("Correcting unwrap errors...")
        disp = correct_region_offset(disp, gnss_los, mask)

        if isinstance(disp, np.ma.MaskedArray):
            disp = disp.filled(np.nan)
        if isinstance(gnss_los, np.ma.MaskedArray):
            gnss_los = gnss_los.filled(np.nan)

        # Downsample if requested
        original_shape = disp.shape
        if self.config.grid_settings.downsample_factor > 1:
            logger.debug(
                f"Downsampling by factor {self.config.grid_settings.downsample_factor}"
            )
            disp_ds = downsample_array(
                disp, self.config.grid_settings.downsample_factor
            )
            gnss_los_ds = downsample_array(
                gnss_los, self.config.grid_settings.downsample_factor
            )
            win_x_ds = max(
                1, window_size_x // self.config.grid_settings.downsample_factor
            )
            win_y_ds = max(
                1, window_size_y // self.config.grid_settings.downsample_factor
            )
        else:
            disp_ds = disp
            gnss_los_ds = gnss_los
            win_x_ds = window_size_x
            win_y_ds = window_size_y

        # Fit calibration surface
        logger.debug("Fitting calibration surface...")
        calibration_surface = self.spatial_processor.fit_windowed_surface(
            disp_ds,
            gnss_los_ds,
            bounds=bounds,
            window_size_x=win_x_ds,
            window_size_y=win_y_ds,
            window_overlap_x=10,
            window_overlap_y=10,
            poly_order=2,
            n_jobs=-1,
        )

        # Apply calibration
        corrected_ds = disp_ds - calibration_surface

        # Upsample if needed
        if self.config.grid_settings.downsample_factor > 1:
            corrected = upsample_array(corrected_ds, original_shape)
        else:
            corrected = corrected_ds

        # Convert back to meters
        corrected = corrected / 1000.0

        # Build output filename
        grid_type = self.config.grid_settings.grid_type
        ref_frame = self.config.grid_settings.reference_frame.lower()
        suffix = f"_corrected_{grid_type}_{ref_frame}"
        if tropo_file is not None:
            suffix += "_tropo"
        if self.config.grid_settings.downsample_factor > 1:
            suffix += f"_downsample{self.config.grid_settings.downsample_factor}"

        output_file = (
            self.config.input_options.work_directory / f"{disp_file.stem}{suffix}.tif"
        )

        # Save
        description = (
            f"Calibrated displacement ({self.config.grid_settings.grid_type} GNSS"
            " model)"
        )
        if tropo_file is not None:
            description += " with tropospheric correction"

        self.io_writer.write_geotiff(
            corrected,
            output_file,
            reference_file=disp_file,
            nodata=np.nan,
            descriptions=[description],
        )

        logger.debug(f"Saved: {output_file.name}")
        return output_file

    def run(self) -> CalibrationState:
        """Run the complete calibration workflow.

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

        # Get bounds
        bounds = self.io_reader.get_bounds(disp_files[0], as_latlon=False)

        # Process each file
        for tropo_file, disp_file in tqdm(self.matched_files, desc="Calibrating"):
            # Handle "None" string from match_correction_to_displacement
            tropo_path = None if tropo_file == "None" else Path(tropo_file)

            output_file = self.process_displacement_file(
                disp_file,
                los_east,
                los_north,
                los_up,
                mask,
                ref_point,
                window_size_pixels,
                window_size_pixels,
                bounds,
                tropo_file=tropo_path,
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
        logger.info(f"Output directory: {self.config.input_options.work_directory}")

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
