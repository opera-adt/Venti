"""Main CLI entry point for Venti.

This module provides the command-line interface for the Venti calibration workflow.

Usage:
    python -m venti config <output_file>    # Generate config template
    python -m venti run <config_file>       # Run calibration workflow
"""

from __future__ import annotations

import logging
import sys
from enum import Enum

import tyro

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Command(str, Enum):
    """Available Venti commands."""

    config = "config"
    run = "run"
    run_single = "run-single"


def config_command(output_dir: str = ".") -> None:
    """Generate configuration file templates.

    Generates two files:
    - runconfig.yaml: Run-specific settings with BOTH calibration and decomposition
      input groups (fill in the one you need based on workflow type)
    - algorithm_parameters.yaml: Algorithm defaults (GNSS, decomposition settings)

    The runconfig.yaml is structured similar to algorithm_parameters.yaml:
    - calibration_input_group section for calibration workflow
    - decomposition_input_group section for decomposition workflow
    - Users only fill in the section they need based on workflow_name

    Parameters
    ----------
    output_dir : str
        Directory to write configuration files (default: current directory)

    Examples
    --------
    ::

        # Generate config files in current directory
        python -m venti config

        # Generate config files in specific directory
        python -m venti config --output-dir configs/

    """
    from .workflow.config import create_config_templates

    try:
        runconfig_path, params_path = create_config_templates(output_dir)
        logger.info("Configuration templates created:")
        logger.info(f"  1. {runconfig_path}")
        logger.info(f"  2. {params_path}")
        logger.info("")
        logger.info("The runconfig.yaml contains TWO input group sections:")
        logger.info("  - calibration_input_group (for calibration workflow)")
        logger.info("  - decomposition_input_group (for decomposition workflow)")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Edit runconfig.yaml:")
        logger.info("     - Fill in calibration_input_group if doing calibration")
        logger.info("     - Fill in decomposition_input_group if doing decomposition")
        logger.info("     - Set workflow_name to 'calibrate' or 'decompose'")
        logger.info("  2. Review algorithm_parameters.yaml (defaults usually work)")
        logger.info("  3. Run: python -m venti run runconfig.yaml")
    except Exception:
        logger.exception("Failed to create configuration templates")
        sys.exit(1)


def run_command(config_file: str, log_level: str = "INFO") -> None:
    """Run the Venti calibration workflow.

    Parameters
    ----------
    config_file : str
        Path to YAML configuration file
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR), default: INFO

    Examples
    --------
    ::

        python -m venti run config.yaml
        python -m venti run config.yaml --log-level DEBUG

    """
    from .workflow.calibration import CalibrationWorkflow
    from .workflow.config import load_config

    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger().setLevel(numeric_level)

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {config_file}")
        config = load_config(config_file)

        logger.info("Configuration loaded successfully")
        logger.info(f"  Input directory: {config.input_options.input_files}")
        logger.info(
            f"  Output directory: {config.run_config.product_path_group.product_path}"
        )
        logger.info(f"  Grid type: {config.grid_settings.grid_type}")
        logger.info(f"  Reference frame: {config.grid_settings.reference_frame}")

        # Run workflow using OO API
        logger.info("Starting calibration workflow...")
        workflow = CalibrationWorkflow(config=config)
        workflow.run()

        logger.info("Workflow completed successfully!")

    except FileNotFoundError:
        logger.exception("File not found")
        sys.exit(1)
    except ValueError:
        logger.exception("Configuration error")
        sys.exit(1)
    except Exception:
        logger.exception("Workflow failed")
        sys.exit(1)


def run_single_command(
    config_file: str,
    disp_file: str,
    tropo_file: str | None = None,
    log_level: str = "INFO",
) -> None:
    """Calibrate a single displacement file.

    Parameters
    ----------
    config_file : str
        Path to YAML configuration file.
    disp_file : str
        Path to the NetCDF displacement file to calibrate.
    tropo_file : str, optional
        Path to a tropospheric correction GeoTIFF for this epoch.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR), default: INFO.

    Examples
    --------
    ::

        venti run-single runconfig.yaml /data/disp/epoch_001.nc
        venti run-single runconfig.yaml /data/disp/epoch_001.nc \
            --tropo-file /data/tropo/tropo_001.tif
        venti run-single runconfig.yaml /data/disp/epoch_001.nc --log-level DEBUG

    """
    from pathlib import Path

    from .workflow.calibration import CalibrationWorkflow
    from .workflow.config import load_config

    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger().setLevel(numeric_level)

    try:
        logger.info(f"Loading configuration from: {config_file}")
        config = load_config(config_file)

        logger.info("Starting single-file calibration...")
        workflow = CalibrationWorkflow(config=config)
        state = workflow.run_single(
            disp_file=Path(disp_file),
            tropo_file=Path(tropo_file) if tropo_file is not None else None,
        )

        if state.n_files_failed:
            logger.error("Calibration failed — check logs above.")
            sys.exit(1)

        logger.info(f"Output: {state.output_files[0]}")

    except FileNotFoundError:
        logger.exception("File not found")
        sys.exit(1)
    except ValueError:
        logger.exception("Configuration error")
        sys.exit(1)
    except Exception:
        logger.exception("Workflow failed")
        sys.exit(1)


def main() -> None:
    """Run the main CLI with subcommands."""
    # Use tyro for CLI parsing with subcommands
    tyro.extras.subcommand_cli_from_dict(
        {
            Command.config: config_command,
            Command.run: run_command,
            Command.run_single: run_single_command,
        }
    )


if __name__ == "__main__":
    main()
