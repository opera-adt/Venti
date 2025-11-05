"""Workflow module for InSAR displacement processing.

This module provides workflows for InSAR displacement products:

1. Calibration: Using GNSS reference data to calibrate displacement products
2. Decomposition: Converting LOS displacement to ENU components

APIs:
- Configuration management (YAML-based)
- Object-oriented API: CalibrationWorkflow, DecompositionWorkflow classes
- Functional API: run_workflow, calibrate_timeseries, decompose_timeseries functions
"""

from __future__ import annotations

__all__ = [
    # Configuration
    'WorkflowConfig',
    'load_config',
    'create_config_template',
    # Workflow types
    'WorkflowType',
    # Calibration API
    'CalibrationWorkflow',
    'CalibrationState',
    'run_calibration_workflow',
    # Decomposition API
    'DecompositionWorkflow',
    'DecompositionState',
    'run_decomposition_workflow',
    # Functional API
    'run_workflow',
    'calibrate_timeseries',
    'decompose_timeseries',
]

# Lazy imports to avoid heavy dependencies on import
def __getattr__(name: str):
    """Lazily import workflow classes and functions on first access."""
    # Configuration
    if name == 'WorkflowConfig':
        from .config import WorkflowConfig
        globals()['WorkflowConfig'] = WorkflowConfig
        return WorkflowConfig

    if name == 'load_config':
        from .config import load_config
        globals()['load_config'] = load_config
        return load_config

    if name == 'create_config_template':
        from .config import create_config_template
        globals()['create_config_template'] = create_config_template
        return create_config_template

    # Object-oriented API
    if name in ['CalibrationWorkflow', 'CalibrationState', 'run_calibration_workflow']:
        from .calibration import (
            CalibrationWorkflow,
            CalibrationState,
            run_calibration_workflow,
        )

        globals()['CalibrationWorkflow'] = CalibrationWorkflow
        globals()['CalibrationState'] = CalibrationState
        globals()['run_calibration_workflow'] = run_calibration_workflow

        return globals()[name]

    # Decomposition API
    if name in ['DecompositionWorkflow', 'DecompositionState', 'run_decomposition_workflow']:
        from .decomposition import (
            DecompositionWorkflow,
            DecompositionState,
            run_decomposition_workflow,
        )

        globals()['DecompositionWorkflow'] = DecompositionWorkflow
        globals()['DecompositionState'] = DecompositionState
        globals()['run_decomposition_workflow'] = run_decomposition_workflow

        return globals()[name]

    # Functional API
    if name == 'WorkflowType':
        from .run import WorkflowType
        globals()['WorkflowType'] = WorkflowType
        return WorkflowType

    if name == 'run_workflow':
        from .run import run_workflow
        globals()['run_workflow'] = run_workflow
        return run_workflow

    if name == 'calibrate_timeseries':
        from .run import calibrate_timeseries
        globals()['calibrate_timeseries'] = calibrate_timeseries
        return calibrate_timeseries

    if name == 'decompose_timeseries':
        from .run import decompose_timeseries
        globals()['decompose_timeseries'] = decompose_timeseries
        return decompose_timeseries

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
