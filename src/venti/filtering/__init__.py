"""Signal filtering utilities (Gaussian, moving-window plane fitting)."""

from __future__ import annotations

from .gaussian import apply_gaussian
from .low_pass_filters import gaussian_fft, gaussian_spatial, hanning_fft, savitzky_golay
from .moving_window import fit_windowed_plane

__all__ = [
    "apply_gaussian",
    "fit_windowed_plane",
    "gaussian_fft",
    "gaussian_spatial",
    "hanning_fft",
    "savitzky_golay",
]
