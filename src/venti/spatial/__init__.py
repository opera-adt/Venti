"""Spatial calibration module for InSAR displacement products.

Provides windowed polynomial plane fitting to remove long-wavelength
orbital and tropospheric signals using GNSS as a reference surface.
"""

from __future__ import annotations

from .processor import SpatialProcessor

__all__ = ["SpatialProcessor"]
