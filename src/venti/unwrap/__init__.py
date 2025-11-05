"""Unwrap error correction module.

This module provides tools for correcting residual offsets and errors
in unwrapped interferometric phase data.
"""

from .unwrap_corrections import (
    UnwrapCorrector,
    correct_region_offset,
    read_data,
    read_netcdf,
)

__all__ = ["UnwrapCorrector", "correct_region_offset", "read_data", "read_netcdf"]
