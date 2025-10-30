"""
Unwrap error correction module.

This module provides tools for correcting residual offsets and errors
in unwrapped interferometric phase data.
"""
from .unwrap_corrections import (
    UnwrapCorrector,
    read_data,
    read_netcdf,
    correct_region_offset
)

__all__ = [
    'UnwrapCorrector',
    'read_data',
    'read_netcdf',
    'correct_region_offset'
]
