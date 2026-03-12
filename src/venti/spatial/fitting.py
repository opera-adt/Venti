"""Compatibility shim — code has moved to dedicated submodules.

All names are re-exported from their new canonical locations so that
existing imports (``from venti.spatial.fitting import ...``) continue
to work without modification.

Canonical locations:
- ``_weighted_lscov``       -> :mod:`venti.solver.lscov`
- ``_design_matrix_poly``   -> :mod:`venti.solver.design_matrix`
- ``_POLY_N_COEFF``         -> :mod:`venti.solver.design_matrix`
- ``_calc_plane_values``    -> :mod:`venti.solver.plane_fitting`
- ``_calc_plane_uncertainty``-> :mod:`venti.solver.plane_fitting`
- ``_fit_plane``            -> :mod:`venti.solver.plane_fitting`
- ``_fill_gaps``            -> :mod:`venti.spatial.gap_filling`
- ``_fill_gaps_gdal``       -> :mod:`venti.spatial.gap_filling`
- ``_fill_gaps_scipy``      -> :mod:`venti.spatial.gap_filling`
- ``_get_residual_mask``    -> :mod:`venti.spatial.gap_filling`
- ``_find_data_extent``     -> :mod:`venti.filtering.moving_window`
- ``_get_sliding_windows``  -> :mod:`venti.filtering.moving_window`
- ``_extend_window``        -> :mod:`venti.filtering.moving_window`
- ``_get_coordinate_grid``  -> :mod:`venti.filtering.moving_window`
- ``_process_window``       -> :mod:`venti.filtering.moving_window`
- ``fit_windowed_plane``    -> :mod:`venti.filtering.moving_window`
"""

from __future__ import annotations

from ..filtering.moving_window import (
    _extend_window,
    _find_data_extent,
    _get_coordinate_grid,
    _get_sliding_windows,
    _process_window,
    fit_windowed_plane,
)
from ..solver.design_matrix import _POLY_N_COEFF, _design_matrix_poly
from ..solver.lscov import _weighted_lscov
from ..solver.plane_fitting import (
    _calc_plane_uncertainty,
    _calc_plane_values,
    _fit_plane,
)
from .gap_filling import (
    _fill_gaps,
    _fill_gaps_gdal,
    _fill_gaps_scipy,
    _get_residual_mask,
)

__all__ = [
    "_weighted_lscov",
    "_design_matrix_poly",
    "_POLY_N_COEFF",
    "_calc_plane_values",
    "_calc_plane_uncertainty",
    "_fit_plane",
    "_fill_gaps",
    "_fill_gaps_gdal",
    "_fill_gaps_scipy",
    "_get_residual_mask",
    "_find_data_extent",
    "_get_sliding_windows",
    "_extend_window",
    "_get_coordinate_grid",
    "_process_window",
    "fit_windowed_plane",
]
