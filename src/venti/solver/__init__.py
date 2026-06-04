"""Numerical solvers for least-squares surface fitting."""

from __future__ import annotations

from .design_matrix import _POLY_N_COEFF, _design_matrix_poly
from .lscov import _weighted_lscov
from .plane_fitting import _calc_plane_uncertainty, _calc_plane_values, _fit_plane

__all__ = [
    "_POLY_N_COEFF",
    "_calc_plane_uncertainty",
    "_calc_plane_values",
    "_design_matrix_poly",
    "_fit_plane",
    "_weighted_lscov",
]
