"""Spatial calibration module for InSAR displacement products.

Provides windowed polynomial plane fitting, gap filling, interpolation,
and resampling utilities.
"""

from __future__ import annotations

from .gap_filling import _fill_gaps, _get_residual_mask
from .interpolation import interpolate_griddata, interpolate_rbf
from .processor import SpatialProcessor
from .resample import downsample_array, upsample_array

__all__ = [
    "SpatialProcessor",
    "_fill_gaps",
    "_get_residual_mask",
    "downsample_array",
    "interpolate_griddata",
    "interpolate_rbf",
    "upsample_array",
]
