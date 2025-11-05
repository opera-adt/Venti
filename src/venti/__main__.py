"""Main CLI entry point for Venti.

This module provides the command-line interface for the Venti calibration workflow.

Usage:
    python -m venti config <output_file>    # Generate config template
    python -m venti run <config_file>       # Run calibration workflow
"""

from __future__ import annotations

import sys
import logging
import tyro
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Command(str, Enum):
    """Available Venti commands."""
    config = "config"
    run = "run"


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
    except Exception as e:
        logger.error(f"Failed to create configuration templates: {e}")
        sys.exit(1)


def run_command(
    config_file: str,
    log_level: str = "INFO"
) -> None:
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
    from .workflow.config import load_config
    from .workflow.calibration import CalibrationWorkflow

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
        logger.info(f"  Output directory: {config.input_options.work_directory}")
        logger.info(f"  Grid type: {config.grid_settings.grid_type}")
        logger.info(f"  Reference frame: {config.grid_settings.reference_frame}")

        # Run workflow using OO API
        logger.info("Starting calibration workflow...")
        workflow = CalibrationWorkflow(config=config)
        state = workflow.run()

        logger.info("Workflow completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point with subcommands."""
    # Use tyro for CLI parsing with subcommands
    tyro.extras.subcommand_cli_from_dict({
        Command.config: config_command,
        Command.run: run_command,
    })


if __name__ == "__main__":
    main()
