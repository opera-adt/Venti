"""Signal filtering utilities (Gaussian, moving-window plane fitting)."""

from __future__ import annotations

from .gaussian import apply_gaussian
from .moving_window import fit_windowed_plane

__all__ = [
    "apply_gaussian",
    "fit_windowed_plane",
]
