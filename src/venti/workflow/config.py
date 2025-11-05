"""Configuration handling for Venti workflows.

This module provides configuration management split into two files:
1. runconfig.yaml: Inputs, outputs, product version (changes per run)
2. algorithm_parameters.yaml: Algorithm settings with defaults (rarely changes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
import yaml


# ============================================================================
# Algorithm Parameters (algorithm_parameters.yaml)
# ============================================================================

class CalibrationOptions(BaseModel):
    """Calibration algorithm options.

    Attributes
    ----------
    grid_type : str
        GNSS grid type: constant (velocity-based) or variable (epoch-specific)
    reference_frame : str
        GNSS reference frame (IGS14 or IGS20)
    starting_year : float
        Starting year for velocity estimation
    unwrap_error_correction : bool
        Whether to correct islands for unwrap errors
    downsample_factor : int
        Downsample factor for plane fitting
    window_size_meters : float
        Window size for plane fitting in meters
    posting_meters : float
        Input data posting in meters
    """

    grid_type: Literal["constant", "variable"] = Field(
        "constant",
        description="GNSS grid type: 'constant' uses velocity-based interpolation, "
                   "'variable' uses epoch-specific GNSS positions"
    )
    reference_frame: str = Field(
        "IGS20",
        description="GNSS reference frame (IGS14 or IGS20)"
    )
    starting_year: float = Field(
        2014.0,
        description="Starting year for velocity estimation (for constant grid type)"
    )
    unwrap_error_correction: bool = Field(
        True,
        description="Whether to correct islands for unwrap errors using watershed segmentation"
    )
    downsample_factor: int = Field(
        1,
        ge=1,
        description="Downsample factor for performing plane fitting at lower resolution"
    )
    window_size_meters: float = Field(
        30000.0,
        gt=0,
        description="Window size for plane fitting in meters"
    )
    posting_meters: float = Field(
        30.0,
        gt=0,
        description="Input data posting (pixel spacing) in meters"
    )


class DecompositionOptions(BaseModel):
    """Decomposition algorithm options.

    Attributes
    ----------
    inversion_method : str
        Method for least-squares inversion
    uncertainty_propagation : bool
        Whether to compute and propagate uncertainties
    min_geometries : int
        Minimum number of viewing geometries required
    vertical_only : bool
        Only solve for vertical component (ignore horizontal)
    geometry_weighting : str
        How to weight different geometries
    """

    inversion_method: Literal["weighted_least_squares", "ridge", "lasso"] = Field(
        "weighted_least_squares",
        description="Method for least-squares inversion"
    )
    uncertainty_propagation: bool = Field(
        True,
        description="Whether to compute and propagate uncertainties to output"
    )
    min_geometries: int = Field(
        2,
        ge=1,
        description="Minimum number of viewing geometries required for decomposition"
    )
    vertical_only: bool = Field(
        False,
        description="Only solve for vertical component (assume horizontal motion is zero)"
    )
    geometry_weighting: Literal["uniform", "temporal_coherence", "uncertainty"] = Field(
        "temporal_coherence",
        description="How to weight different viewing geometries in inversion"
    )
    quality_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum quality metric for including pixels in decomposition"
    )


class OutputOptions(BaseModel):
    """Output file options.

    Attributes
    ----------
    gtiff_creation_options : list
        GDAL creation options for GeoTIFF files
    add_overviews : bool
        Whether to add overviews to output GeoTIFFs
    compression : str
        Compression method for output files
    """

    gtiff_creation_options: list[str] = Field(
        default_factory=lambda: [
            "COMPRESS=lzw",
            "ZLEVEL=4",
            "BIGTIFF=yes",
            "TILED=yes",
            "BLOCKXSIZE=128",
            "BLOCKYSIZE=128",
        ],
        description="GDAL creation options for GeoTIFF output files"
    )
    add_overviews: bool = Field(
        False,
        description="Whether to add overviews (pyramids) to output GeoTIFFs"
    )
    compression: str = Field(
        "lzw",
        description="Compression method for output files"
    )


class AlgorithmParameters(BaseModel):
    """Algorithm parameters configuration.

    This contains algorithm-specific settings that typically don't change
    between runs. These are saved in algorithm_parameters.yaml.
    """

    calibration_options: CalibrationOptions = Field(
        default_factory=CalibrationOptions,
        description="Settings for calibration workflow"
    )
    decomposition_options: DecompositionOptions = Field(
        default_factory=DecompositionOptions,
        description="Settings for decomposition workflow"
    )
    output_options: OutputOptions = Field(
        default_factory=OutputOptions,
        description="Output file format options"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> AlgorithmParameters:
        """Load algorithm parameters from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Algorithm parameters file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save algorithm parameters to YAML file."""
        yaml_path = Path(yaml_path)
        data = self.model_dump(mode='python')

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# Run Configuration (runconfig.yaml)
# ============================================================================

class CalibrationInputGroup(BaseModel):
    """Input file group for calibration workflow.

    Attributes
    ----------
    input_files : Path
        Directory containing input NetCDF displacement files
    los_file : Path
        Path to LOS unit vector file (3-band GeoTIFF: east, north, up)
    water_mask : Path
        Path to water mask file (GeoTIFF)
    tropo_files : Path, optional
        Directory with tropospheric correction files
    reference_point : tuple[int, int], optional
        Reference point (row, col), None for auto-select
    """

    input_files: Path = Field(
        ...,
        description="Directory containing input NetCDF displacement files from OPERA DISP products"
    )
    los_file: Path = Field(
        ...,
        description="Path to LOS unit vector file (3-band GeoTIFF: east, north, up components)"
    )
    water_mask: Path = Field(
        ...,
        description="Path to water mask file (GeoTIFF, 1=valid land, 0=invalid/water)"
    )
    tropo_files: Optional[Path] = Field(
        None,
        description="Directory with tropospheric correction NetCDF files (optional)"
    )
    reference_point: Optional[tuple[int, int]] = Field(
        None,
        description="Reference point as [row, col] in pixel coordinates, None for automatic selection"
    )

    @field_validator('input_files', 'los_file', 'water_mask', mode='before')
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if v is not None:
            return Path(v)
        return v

    @field_validator('tropo_files', mode='before')
    @classmethod
    def convert_optional_to_path(cls, v):
        """Convert optional string paths to Path objects."""
        if v is not None and v != "":
            return Path(v)
        return None if v == "" else v

    @field_validator('input_files', 'los_file', 'water_mask')
    @classmethod
    def validate_exists(cls, v, info):
        """Validate that required paths exist (skip for template placeholders)."""
        # Skip validation for template placeholder paths
        if str(v).startswith('path/to/'):
            return v
        if not v.exists():
            raise ValueError(f"{info.field_name} does not exist: {v}")
        return v

    @field_validator('tropo_files')
    @classmethod
    def validate_optional_exists(cls, v):
        """Validate that optional paths exist if provided (skip for template placeholders)."""
        if v is None:
            return v
        # Skip validation for template placeholder paths
        if str(v).startswith('path/to/'):
            return v
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v


class DecompositionInputGroup(BaseModel):
    """Input file group for decomposition workflow.

    Attributes
    ----------
    asc_displacement_files : Path
        Directory with ascending geometry displacement files (NetCDF format)
    desc_displacement_files : Path
        Directory with descending geometry displacement files (NetCDF format)
    asc_los_file : Path
        Path to ascending LOS unit vector file (3-band GeoTIFF: east, north, up)
    desc_los_file : Path
        Path to descending LOS unit vector file (3-band GeoTIFF: east, north, up)
    water_mask : Path
        Path to water mask file (GeoTIFF)
    asc_static_layers : Path, optional
        Path to ascending static layers file (GeoTIFF with temporal coherence, etc.)
    desc_static_layers : Path, optional
        Path to descending static layers file (GeoTIFF with temporal coherence, etc.)
    reference_point : tuple[int, int], optional
        Reference point (row, col), None for auto-select
    """

    asc_displacement_files: Path = Field(
        ...,
        description="Directory with ascending geometry displacement files in NetCDF format"
    )
    desc_displacement_files: Path = Field(
        ...,
        description="Directory with descending geometry displacement files in NetCDF format"
    )
    asc_los_file: Path = Field(
        ...,
        description="Path to ascending LOS unit vector file (3-band GeoTIFF: east, north, up)"
    )
    desc_los_file: Path = Field(
        ...,
        description="Path to descending LOS unit vector file (3-band GeoTIFF: east, north, up)"
    )
    water_mask: Path = Field(
        ...,
        description="Path to water mask file (GeoTIFF, 1=valid land, 0=invalid/water)"
    )
    asc_static_layers: Optional[Path] = Field(
        None,
        description="Path to ascending static layers GeoTIFF (temporal coherence, amplitude dispersion, etc.)"
    )
    desc_static_layers: Optional[Path] = Field(
        None,
        description="Path to descending static layers GeoTIFF (temporal coherence, amplitude dispersion, etc.)"
    )
    reference_point: Optional[tuple[int, int]] = Field(
        None,
        description="Reference point as [row, col] in pixel coordinates, None for automatic selection"
    )

    @field_validator(
        'asc_displacement_files',
        'desc_displacement_files',
        'asc_los_file',
        'desc_los_file',
        'water_mask',
        mode='before'
    )
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if v is not None:
            return Path(v)
        return v

    @field_validator('asc_static_layers', 'desc_static_layers', mode='before')
    @classmethod
    def convert_optional_to_path(cls, v):
        """Convert optional string paths to Path objects."""
        if v is not None and v != "":
            return Path(v)
        return None if v == "" else v

    @field_validator(
        'asc_displacement_files',
        'desc_displacement_files',
        'asc_los_file',
        'desc_los_file',
        'water_mask'
    )
    @classmethod
    def validate_exists(cls, v, info):
        """Validate that required paths exist (skip for template placeholders)."""
        # Skip validation for template placeholder paths
        if str(v).startswith('path/to/'):
            return v
        if not v.exists():
            raise ValueError(f"{info.field_name} does not exist: {v}")
        return v

    @field_validator('asc_static_layers', 'desc_static_layers')
    @classmethod
    def validate_optional_exists(cls, v):
        """Validate that optional paths exist if provided (skip for template placeholders)."""
        if v is None:
            return v
        # Skip validation for template placeholder paths
        if str(v).startswith('path/to/'):
            return v
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v


# Type alias for backward compatibility
InputFileGroup = CalibrationInputGroup


class ProductPathGroup(BaseModel):
    """Product path group configuration.

    Attributes
    ----------
    product_path : Path
        Directory where products will be placed
    scratch_path : Path
        Path to scratch directory for intermediate files
    sas_output_path : Path
        Path to SAS output directory
    product_version : str
        Version of the product in <major>.<minor> format
    """

    product_path: Path = Field(
        Path("output"),
        description="Directory where products will be placed"
    )
    scratch_path: Path = Field(
        Path("scratch"),
        description="Path to scratch directory for intermediate files"
    )
    sas_output_path: Path = Field(
        Path("output"),
        description="Path to SAS output directory"
    )
    product_version: str = Field(
        "1.0",
        description="Version of the product in <major>.<minor> format"
    )

    @field_validator('product_path', 'scratch_path', 'sas_output_path', mode='before')
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class WorkerSettings(BaseModel):
    """Worker configuration for processing.

    Attributes
    ----------
    gpu_enabled : bool
        Whether to use GPU for processing (if available)
    threads_per_worker : int
        Number of threads to use per worker (sets OMP_NUM_THREADS)
    block_shape : list[int]
        Size (rows, columns) of blocks of data to load at a time
    """

    gpu_enabled: bool = Field(
        False,
        description="Whether to use GPU for processing (if available)"
    )
    threads_per_worker: int = Field(
        1,
        ge=1,
        description="Number of threads to use per worker (sets OMP_NUM_THREADS)"
    )
    block_shape: list[int] = Field(
        [512, 512],
        min_length=2,
        max_length=2,
        description="Size (rows, columns) of blocks of data to load at a time"
    )

    @field_validator('block_shape')
    @classmethod
    def validate_block_shape(cls, v):
        """Validate block shape dimensions."""
        if len(v) != 2:
            raise ValueError("block_shape must have exactly 2 elements")
        if any(dim <= 0 for dim in v):
            raise ValueError("block_shape dimensions must be positive")
        return v


class PrimaryExecutable(BaseModel):
    """Primary executable configuration.

    Attributes
    ----------
    product_type : str
        Product type of the workflow
    workflow_name : str
        Name of the workflow to execute
    """

    product_type: Literal["VENTI_CALIBRATION", "VENTI_DECOMPOSITION"] = Field(
        "VENTI_CALIBRATION",
        description="Product type of the workflow"
    )
    workflow_name: Literal["calibrate", "decompose"] = Field(
        "calibrate",
        description="Name of the workflow to execute"
    )


class RunConfig(BaseModel):
    """Run configuration.

    This contains run-specific settings that change between runs.
    These are saved in runconfig.yaml.

    Note: Provide either calibration_input_group OR decomposition_input_group
    based on the workflow type.
    """

    calibration_input_group: Optional[CalibrationInputGroup] = Field(
        None,
        description="Input files for calibration workflow (required if workflow_name is 'calibrate')"
    )
    decomposition_input_group: Optional[DecompositionInputGroup] = Field(
        None,
        description="Input files for decomposition workflow (required if workflow_name is 'decompose')"
    )
    product_path_group: ProductPathGroup = Field(
        default_factory=ProductPathGroup,
        description="Output product paths and version"
    )
    primary_executable: PrimaryExecutable = Field(
        default_factory=PrimaryExecutable,
        description="Primary executable configuration"
    )
    worker_settings: WorkerSettings = Field(
        default_factory=WorkerSettings,
        description="Worker configuration for processing"
    )
    log_file: Optional[str] = Field(
        None,
        description="Path to output log file (in addition to logging to stderr)"
    )
    keep_paths_relative: bool = Field(
        False,
        description="Don't resolve filepaths that are given as relative to be absolute"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }

    def model_post_init(self, __context) -> None:
        """Validate that the correct input group is provided for the workflow type."""
        workflow_name = self.primary_executable.workflow_name

        if workflow_name == 'calibrate':
            if self.calibration_input_group is None:
                raise ValueError(
                    "calibration_input_group is required when workflow_name is 'calibrate'. "
                    "Fill in the calibration_input_group section with your data paths."
                )
        elif workflow_name == 'decompose':
            if self.decomposition_input_group is None:
                raise ValueError(
                    "decomposition_input_group is required when workflow_name is 'decompose'. "
                    "Fill in the decomposition_input_group section with your data paths."
                )

    @property
    def input_file_group(self):
        """Get the active input file group based on workflow type."""
        if self.primary_executable.workflow_name == 'calibrate':
            return self.calibration_input_group
        else:
            return self.decomposition_input_group

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> RunConfig:
        """Load run configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Run configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save run configuration to YAML file."""
        yaml_path = Path(yaml_path)
        data = self.model_dump(mode='python')

        # Convert Path objects to strings for YAML serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = convert_paths(data)

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# Combined Configuration
# ============================================================================

class VentiConfig(BaseModel):
    """Combined Venti configuration.

    This combines both run configuration and algorithm parameters.
    """

    run_config: RunConfig
    algorithm_parameters: AlgorithmParameters

    @classmethod
    def from_yaml_files(
        cls,
        runconfig_path: str | Path,
        algorithm_params_path: str | Path,
    ) -> VentiConfig:
        """Load configuration from separate YAML files.

        Parameters
        ----------
        runconfig_path : str or Path
            Path to runconfig.yaml file
        algorithm_params_path : str or Path
            Path to algorithm_parameters.yaml file

        Returns
        -------
        VentiConfig
            Combined configuration
        """
        run_config = RunConfig.from_yaml(runconfig_path)
        algorithm_params = AlgorithmParameters.from_yaml(algorithm_params_path)

        return cls(
            run_config=run_config,
            algorithm_parameters=algorithm_params,
        )

    @classmethod
    def from_legacy_yaml(cls, yaml_path: str | Path) -> VentiConfig:
        """Load configuration from legacy single YAML file.

        This provides backward compatibility with the old config format.
        """
        from .config import WorkflowConfig

        # Load old config
        old_config = WorkflowConfig.from_yaml(yaml_path)

        # Convert to new structure
        run_config = RunConfig(
            input_file_group=InputFileGroup(
                input_files=old_config.input_options.input_files,
                los_file=old_config.input_options.los_file,
                water_mask=old_config.input_options.water_mask,
                tropo_files=old_config.input_options.tropo_files,
                reference_point=old_config.input_options.reference_point,
            ),
            product_path_group=ProductPathGroup(
                product_path=old_config.input_options.work_directory,
                scratch_path=old_config.input_options.work_directory / "scratch",
                sas_output_path=old_config.input_options.work_directory,
                product_version="1.0",
            ),
            worker_settings=WorkerSettings(
                gpu_enabled=old_config.worker_settings.gpu_enabled,
                threads_per_worker=old_config.worker_settings.threads_per_worker,
                block_shape=old_config.worker_settings.block_shape,
            ),
            log_file=old_config.log_file,
            keep_paths_relative=old_config.keep_paths_relative,
        )

        algorithm_params = AlgorithmParameters(
            calibration_options=CalibrationOptions(
                grid_type=old_config.grid_settings.grid_type,
                reference_frame=old_config.grid_settings.reference_frame,
                starting_year=old_config.grid_settings.starting_year,
                unwrap_error_correction=old_config.unwrap_error_correction,
                downsample_factor=old_config.grid_settings.downsample_factor,
                window_size_meters=old_config.grid_settings.window_size_meters,
                posting_meters=old_config.grid_settings.posting_meters,
            ),
        )

        return cls(
            run_config=run_config,
            algorithm_parameters=algorithm_params,
        )


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(
    runconfig_path: str | Path,
    algorithm_params_path: Optional[str | Path] = None,
) -> VentiConfig:
    """Load Venti configuration from YAML files.

    Parameters
    ----------
    runconfig_path : str or Path
        Path to runconfig.yaml file
    algorithm_params_path : str or Path, optional
        Path to algorithm_parameters.yaml file.
        If None, looks for algorithm_parameters.yaml in same directory as runconfig.

    Returns
    -------
    VentiConfig
        Combined configuration
    """
    runconfig_path = Path(runconfig_path)

    if algorithm_params_path is None:
        # Look for algorithm_parameters.yaml in same directory
        algorithm_params_path = runconfig_path.parent / "algorithm_parameters.yaml"

        if not algorithm_params_path.exists():
            # Try legacy single-file config
            return VentiConfig.from_legacy_yaml(runconfig_path)

    return VentiConfig.from_yaml_files(runconfig_path, algorithm_params_path)


def _write_yaml_with_comments(data: dict, model: type[BaseModel], output_path: Path) -> None:
    """Write YAML with inline comments from Field descriptions.

    Parameters
    ----------
    data : dict
        Data to write
    model : type[BaseModel]
        Pydantic model class to extract descriptions from
    output_path : Path
        Path to write YAML file
    """
    def write_dict_with_comments(obj: dict, model_class: type[BaseModel], indent: int = 0) -> str:
        """Recursively write dict with comments."""
        lines = []
        indent_str = "  " * indent

        for key, value in obj.items():
            # Get field description if available
            if hasattr(model_class, 'model_fields') and key in model_class.model_fields:
                field_info = model_class.model_fields[key]
                description = field_info.description
                if description:
                    # Write comment above field
                    lines.append(f"{indent_str}# {description}")

            # Handle nested dicts (nested models)
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                # Get nested model class if available
                if hasattr(model_class, 'model_fields') and key in model_class.model_fields:
                    field_info = model_class.model_fields[key]
                    # Get the actual type (handle Optional, etc.)
                    field_type = field_info.annotation
                    if hasattr(field_type, '__origin__'):  # Handle Optional, Union
                        args = getattr(field_type, '__args__', ())
                        field_type = args[0] if args else field_type
                    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                        lines.append(write_dict_with_comments(value, field_type, indent + 1))
                    else:
                        lines.append(write_dict_with_comments(value, model_class, indent + 1))
                else:
                    lines.append(write_dict_with_comments(value, model_class, indent + 1))
            # Handle lists
            elif isinstance(value, list):
                lines.append(f"{indent_str}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{indent_str}  -")
                        lines.append(write_dict_with_comments(item, model_class, indent + 2))
                    else:
                        lines.append(f"{indent_str}  - {item}")
            # Handle None
            elif value is None:
                lines.append(f"{indent_str}{key}: null")
            # Handle strings with special characters
            elif isinstance(value, str):
                if any(c in value for c in [':', '#', '@', '`']):
                    lines.append(f"{indent_str}{key}: '{value}'")
                else:
                    lines.append(f"{indent_str}{key}: {value}")
            # Handle Path objects
            elif isinstance(value, Path):
                lines.append(f"{indent_str}{key}: {str(value)}")
            # Handle everything else
            else:
                lines.append(f"{indent_str}{key}: {value}")

        return '\n'.join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = write_dict_with_comments(data, model)

    with open(output_path, 'w') as f:
        f.write(content)


def create_config_templates(output_dir: str | Path = ".") -> tuple[Path, Path]:
    """Create template configuration files.

    Generates two template files:
    1. runconfig.yaml - Contains both calibration and decomposition input groups
    2. algorithm_parameters.yaml - Algorithm settings (shared)

    The runconfig.yaml includes both input groups - users only fill in the one
    they need based on their workflow type (calibrate or decompose).

    Parameters
    ----------
    output_dir : str or Path
        Directory to write template files

    Returns
    -------
    tuple of Path
        (runconfig_path, algorithm_params_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runconfig_path = output_dir / "runconfig.yaml"
    algorithm_params_path = output_dir / "algorithm_parameters.yaml"

    # Create config with BOTH input groups as examples
    # Users will fill in only the one they need based on workflow_name
    config = RunConfig(
        calibration_input_group=CalibrationInputGroup(
            input_files=Path("path/to/displacement/files"),
            los_file=Path("path/to/los_vectors.tif"),
            water_mask=Path("path/to/water_mask.tif"),
            tropo_files=None,
            reference_point=None,
        ),
        decomposition_input_group=DecompositionInputGroup(
            asc_displacement_files=Path("path/to/asc_displacement/"),
            desc_displacement_files=Path("path/to/desc_displacement/"),
            asc_los_file=Path("path/to/asc_los_vectors.tif"),
            desc_los_file=Path("path/to/desc_los_vectors.tif"),
            water_mask=Path("path/to/water_mask.tif"),
            asc_static_layers=None,
            desc_static_layers=None,
            reference_point=None,
        ),
        primary_executable=PrimaryExecutable(
            product_type="VENTI_CALIBRATION",
            workflow_name="calibrate",
        ),
    )

    algorithm_params = AlgorithmParameters()

    # Convert configs to dicts
    config_data = config.model_dump(mode='python')
    algorithm_params_data = algorithm_params.model_dump(mode='python')

    # Convert Path objects to strings
    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    config_data = convert_paths(config_data)
    algorithm_params_data = convert_paths(algorithm_params_data)

    # Write YAML files with comments
    _write_yaml_with_comments(config_data, RunConfig, runconfig_path)
    _write_yaml_with_comments(algorithm_params_data, AlgorithmParameters, algorithm_params_path)

    return runconfig_path, algorithm_params_path


# ============================================================================
# Type Alias for Compatibility
# ============================================================================

# Type alias for cleaner imports
WorkflowConfig = VentiConfig

