"""Main workflow runner for InSAR displacement processing.

This module provides workflow execution functions for:
1. Calibration: Calibrating displacement products using GNSS reference data
2. Decomposition: Converting LOS displacement to ENU components

Note: The calibration workflow wraps the CalibrationWorkflow class
defined in calibration.py. It maintains backward compatibility with the functional
API while using the object-oriented implementation under the hood.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from .calibration import CalibrationState, CalibrationWorkflow
from .config import AlgorithmParameters, RunConfig, VentiConfig, WorkflowConfig
from .decomposition import DecompositionState, DecompositionWorkflow

logger = logging.getLogger(__name__)


class WorkflowType(StrEnum):
    """Available workflow types."""

    calibrate = "calibrate"
    decompose = "decompose"


def run_workflow(
    config: WorkflowConfig, workflow_type: WorkflowType = WorkflowType.calibrate
) -> CalibrationState | DecompositionState:
    """Run a workflow using provided configuration.

    Parameters
    ----------
    config : WorkflowConfig
        Configuration object with all workflow parameters
    workflow_type : WorkflowType, optional
        Type of workflow to run: 'calibrate' or 'decompose', by default 'calibrate'

    Returns
    -------
    CalibrationState or DecompositionState
        Final workflow state with processing statistics

    Examples
    --------
    Run calibration workflow::

        from venti.workflow import WorkflowConfig, run_workflow, WorkflowType

        config = WorkflowConfig.from_yaml('config.yaml')
        state = run_workflow(config, WorkflowType.calibrate)
        print(f"Processed {state.n_files_processed} files")

    Run decomposition workflow::

        config = WorkflowConfig.from_yaml('config.yaml')
        state = run_workflow(config, WorkflowType.decompose)

    """
    # Set up worker settings
    import os

    if config.worker_settings.threads_per_worker > 1:
        os.environ["OMP_NUM_THREADS"] = str(config.worker_settings.threads_per_worker)

    # Set up logging if log_file is specified
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(file_handler)

    # Dispatch to appropriate workflow
    if workflow_type == WorkflowType.calibrate:
        calib_workflow = CalibrationWorkflow(config=config)
        return calib_workflow.run()
    elif workflow_type == WorkflowType.decompose:
        decomp_workflow = DecompositionWorkflow(config=config)
        return decomp_workflow.run()
    else:
        msg = f"Unknown workflow type: {workflow_type}"
        raise ValueError(msg)


def calibrate_timeseries(
    input_dir: Path,
    los_file: Path,
    mask_file: Path,
    output_dir: Path,
    correction_dir: Path | None = None,
    reference_point: tuple[int, int] | None = None,
    grid_type: Literal["constant", "variable"] = "constant",
    downsample_factor: int = 1,
    window_size_meters: float = 30000,
    posting_meters: float = 30,
    reference_frame: str = "IGS20",
    start_year: float = 2014.0,
    gpu_enabled: bool = False,
    block_shape: list[int] | None = None,
    unwrap_error_correction: bool = True,
    keep_paths_relative: bool = False,
) -> CalibrationState:
    """Calibrate InSAR displacement timeseries using GNSS reference data.

    This function provides a backward-compatible functional API that wraps
    the CalibrationWorkflow class. For new code, consider using WorkflowConfig
    and run_workflow directly.

    Parameters
    ----------
    input_dir : Path
        Directory containing input NetCDF displacement files
    los_file : Path
        Path to LOS unit vector file (3-band GeoTIFF: east, north, up)
    mask_file : Path
        Path to mask file (1=valid, 0=invalid)
    output_dir : Path
        Output directory for corrected displacements and intermediate products
    correction_dir : Path, optional
        Directory with tropospheric correction files, by default None
    reference_point : tuple of int, optional
        Reference point (row, col), by default None (auto-select)
    grid_type : str, optional
        GNSS grid type: "constant" (velocity) or "variable" (epoch-specific),
        by default "constant"
    downsample_factor : int, optional
        Downsampling factor for faster processing, by default 1 (no downsampling)
    window_size_meters : float, optional
        Window size for surface fitting in meters, by default 30000
    posting_meters : float, optional
        Pixel posting in meters, by default 30
    reference_frame : str, optional
        GNSS reference frame, by default "IGS20"
    start_year : float, optional
        Start year for velocity calculation (if grid_type="constant"), by default 2014.0
    gpu_enabled : bool, optional
        Whether to use GPU for processing (if available), by default False
    block_shape : list of int, optional
        Size (rows, columns) of blocks of data to load at a time, by default [512, 512]
    unwrap_error_correction : bool, optional
        Whether to correct islands for unwrap errors, by default True
    keep_paths_relative : bool, optional
        Don't resolve filepaths that are given as relative to be absolute,
        by default False

    Returns
    -------
    CalibrationState
        Final workflow state with processing statistics

    Examples
    --------
    Calibrate with constant velocity model::

        state = calibrate_timeseries(
            input_dir=Path('displacement/'),
            los_file=Path('los_vectors.tif'),
            mask_file=Path('water_mask.tif'),
            output_dir=Path('calibrated/'),
            grid_type='constant'
        )

    Calibrate with tropospheric corrections::

        state = calibrate_timeseries(
            input_dir=Path('displacement/'),
            los_file=Path('los_vectors.tif'),
            mask_file=Path('water_mask.tif'),
            output_dir=Path('calibrated/'),
            correction_dir=Path('tropo_corrections/'),
            grid_type='constant'
        )

    Notes
    -----
    Output GeoTIFF files are saved with suffix based on processing:
    - _corrected_constant_igs20: Constant velocity model
    - _corrected_variable_igs20: Epoch-specific model
    - _corrected_*_tropo: With tropospheric corrections
    - _corrected_*_downsample{N}: With downsampling

    """
    from .config import (
        CalibrationInputGroup,
        CalibrationOptions,
        PrimaryExecutable,
        ProcessingOptions,
        ProductPathGroup,
        WorkerSettings,
    )

    # Set default block_shape if not provided
    if block_shape is None:
        block_shape = [512, 512]

    # Build configuration from parameters
    run_config = RunConfig(
        calibration_input_group=CalibrationInputGroup(
            input_files=input_dir,
            los_file=los_file,
            water_mask=mask_file,
            tropo_files=correction_dir,
            reference_point=reference_point,
        ),
        product_path_group=ProductPathGroup(
            product_path=output_dir,
            scratch_path=output_dir / "scratch",
            sas_output_path=output_dir,
            product_version="1.0",
        ),
        primary_executable=PrimaryExecutable(
            product_type="CAL",
            workflow_name="calibrate",
        ),
        worker_settings=WorkerSettings(
            gpu_enabled=gpu_enabled,
            threads_per_worker=1,
            block_shape=block_shape,
        ),
        keep_paths_relative=keep_paths_relative,
    )

    algorithm_params = AlgorithmParameters(
        processing_options=ProcessingOptions(
            cal_downsample_factor=downsample_factor,
        ),
        calibration_options=CalibrationOptions(
            grid_type=grid_type,
            reference_frame=reference_frame,
            starting_year=start_year,
            unwrap_error_correction=unwrap_error_correction,
            window_size_meters=window_size_meters,
            posting_meters=posting_meters,
        ),
    )

    config = VentiConfig(
        run_config=run_config,
        algorithm_parameters=algorithm_params,
    )

    # Run workflow using the CalibrationWorkflow class
    return cast(
        CalibrationState, run_workflow(config, workflow_type=WorkflowType.calibrate)
    )


def decompose_timeseries(
    input_dir: Path,
    los_file: Path,
    mask_file: Path,
    output_dir: Path,
    asc_dir: Path | None = None,
    desc_dir: Path | None = None,
    reference_point: tuple[int, int] | None = None,
    gpu_enabled: bool = False,
    block_shape: list[int] | None = None,
    keep_paths_relative: bool = False,
) -> DecompositionState:
    """Decompose InSAR LOS displacement to East-North-Up components.

    This function provides a backward-compatible functional API that wraps
    the DecompositionWorkflow class. For new code, consider using WorkflowConfig
    and run_workflow directly.

    Parameters
    ----------
    input_dir : Path
        Directory containing input NetCDF displacement files
    los_file : Path
        Path to LOS unit vector file (3-band GeoTIFF: east, north, up)
    mask_file : Path
        Path to mask file (1=valid, 0=invalid)
    output_dir : Path
        Output directory for decomposed displacement components
    asc_dir : Path, optional
        Directory with ascending geometry displacement files, by default None
    desc_dir : Path, optional
        Directory with descending geometry displacement files, by default None
    reference_point : tuple of int, optional
        Reference point (row, col), by default None (auto-select)
    gpu_enabled : bool, optional
        Whether to use GPU for processing (if available), by default False
    block_shape : list of int, optional
        Size (rows, columns) of blocks of data to load at a time, by default [512, 512]
    keep_paths_relative : bool, optional
        Don't resolve filepaths that are given as relative to be absolute,
        by default False

    Returns
    -------
    DecompositionState
        Final workflow state with processing statistics

    Examples
    --------
    Decompose with single geometry::

        state = decompose_timeseries(
            input_dir=Path('displacement/'),
            los_file=Path('los_vectors.tif'),
            mask_file=Path('water_mask.tif'),
            output_dir=Path('decomposed/')
        )

    Decompose with ascending and descending::

        state = decompose_timeseries(
            input_dir=Path('displacement/'),
            los_file=Path('los_vectors.tif'),
            mask_file=Path('water_mask.tif'),
            output_dir=Path('decomposed/'),
            asc_dir=Path('ascending/'),
            desc_dir=Path('descending/')
        )

    Notes
    -----
    This is a PLACEHOLDER implementation. Full decomposition requires:
    - Multiple viewing geometries (ascending/descending)
    - Proper least-squares inversion
    - Uncertainty quantification
    - Quality metrics

    """
    from .config import (
        DecompositionInputGroup,
        PrimaryExecutable,
        ProductPathGroup,
        WorkerSettings,
    )

    # Set default block_shape if not provided
    if block_shape is None:
        block_shape = [512, 512]

    # Build configuration from parameters
    # Note: This API is simplified and assumes asc/desc dirs contain both
    # displacement files and LOS files with standard naming
    run_config = RunConfig(
        decomposition_input_group=DecompositionInputGroup(
            asc_displacement_files=asc_dir if asc_dir else input_dir,
            desc_displacement_files=desc_dir if desc_dir else input_dir,
            asc_los_file=los_file,  # Simplified - should be asc-specific
            desc_los_file=los_file,  # Simplified - should be desc-specific
            water_mask=mask_file,
            reference_point=reference_point,
        ),
        product_path_group=ProductPathGroup(
            product_path=output_dir,
            scratch_path=output_dir / "scratch",
            sas_output_path=output_dir,
            product_version="1.0",
        ),
        primary_executable=PrimaryExecutable(
            product_type="VLM",
            workflow_name="decompose",
        ),
        worker_settings=WorkerSettings(
            gpu_enabled=gpu_enabled,
            threads_per_worker=1,
            block_shape=block_shape,
        ),
        keep_paths_relative=keep_paths_relative,
    )

    algorithm_params = AlgorithmParameters()

    config = VentiConfig(
        run_config=run_config,
        algorithm_parameters=algorithm_params,
    )

    # Run workflow using the DecompositionWorkflow class
    return cast(
        DecompositionState, run_workflow(config, workflow_type=WorkflowType.decompose)
    )


def calibrate_command(config_file: str) -> None:
    """Run calibration workflow from YAML config file.

    Parameters
    ----------
    config_file : str
        Path to YAML configuration file

    """
    from .config import load_config

    config = load_config(config_file)
    state = run_workflow(config, workflow_type=WorkflowType.calibrate)
    logger.info(f"Calibration complete! Processed {state.n_files_processed} files")


def decompose_command(config_file: str) -> None:
    """Run decomposition workflow from YAML config file.

    Parameters
    ----------
    config_file : str
        Path to YAML configuration file

    """
    from .config import load_config

    config = load_config(config_file)
    state = run_workflow(config, workflow_type=WorkflowType.decompose)
    logger.info(f"Decomposition complete! Processed {state.n_files_processed} files")


if __name__ == "__main__":
    import tyro

    # Support subcommands: calibrate or decompose
    tyro.extras.subcommand_cli_from_dict(
        {
            WorkflowType.calibrate: calibrate_timeseries,
            WorkflowType.decompose: decompose_timeseries,
        }
    )
