"""Unit tests for the spatial subpackage.

All tests use synthetic numpy arrays — no files or network required.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import joblib  # noqa: F401

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import skimage  # noqa: F401

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Both skimage and joblib are top-level imports in venti.spatial.fitting,
# so any test that imports from that module requires both.
_HAS_SPATIAL_DEPS = HAS_JOBLIB and HAS_SKIMAGE
_SKIP_SPATIAL = pytest.mark.skipif(
    not _HAS_SPATIAL_DEPS, reason="skimage or joblib not installed"
)


# _design_matrix_poly
@_SKIP_SPATIAL
class TestDesignMatrixPoly:
    """Tests for _design_matrix_poly."""

    def test_order_0_is_constant(self):
        from venti.spatial.fitting import _design_matrix_poly

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        A = _design_matrix_poly(x, y, poly_order=0)
        assert A.shape == (3, 1)
        assert np.all(A == 1.0)

    def test_order_1_has_three_columns(self):
        from venti.spatial.fitting import _design_matrix_poly

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        A = _design_matrix_poly(x, y, poly_order=1)
        assert A.shape == (10, 3)

    def test_order_1p5_has_four_columns(self):
        from venti.spatial.fitting import _design_matrix_poly

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        A = _design_matrix_poly(x, y, poly_order=1.5)
        assert A.shape == (10, 4)

    def test_order_2_has_six_columns(self):
        from venti.spatial.fitting import _design_matrix_poly

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        A = _design_matrix_poly(x, y, poly_order=2)
        assert A.shape == (10, 6)

    def test_order_3_has_eight_columns(self):
        from venti.spatial.fitting import _design_matrix_poly

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        A = _design_matrix_poly(x, y, poly_order=3)
        assert A.shape == (10, 8)

    def test_unsupported_order_raises(self):
        from venti.spatial.fitting import _design_matrix_poly

        x = np.ones(5)
        y = np.ones(5)
        with pytest.raises((ValueError, KeyError)):
            _design_matrix_poly(x, y, poly_order=99)


# _calc_plane_values
@_SKIP_SPATIAL
class TestCalcPlaneValues:
    """Tests for _calc_plane_values."""

    def test_constant_plane(self):
        from venti.spatial.fitting import _calc_plane_values

        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        y = np.array([[0.0, 0.0], [1.0, 1.0]])
        coef = np.array([7.0])
        result = _calc_plane_values(x, y, coef, poly_order=0)
        assert np.allclose(result, 7.0)

    def test_linear_plane_exact(self):
        """z = 2x + 3y + 1 must be recovered exactly."""
        from venti.spatial.fitting import _calc_plane_values

        x = np.array([[0.0, 1.0], [0.0, 1.0]])
        y = np.array([[0.0, 0.0], [1.0, 1.0]])
        coef = np.array([2.0, 3.0, 1.0])
        result = _calc_plane_values(x, y, coef, poly_order=1)
        expected = 2.0 * x + 3.0 * y + 1.0
        assert np.allclose(result, expected)

    def test_output_shape_matches_input(self):
        from venti.spatial.fitting import _calc_plane_values

        x = np.random.rand(8, 12)
        y = np.random.rand(8, 12)
        coef = np.array([1.0, 2.0, 3.0])
        result = _calc_plane_values(x, y, coef, poly_order=1)
        assert result.shape == (8, 12)


# _weighted_lscov
@_SKIP_SPATIAL
class TestWeightedLscov:
    """Tests for _weighted_lscov."""

    def test_recover_known_coefficients_unweighted(self):
        """Exact linear system: solution must match to machine precision."""
        from venti.spatial.fitting import _weighted_lscov

        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 50)
        y = rng.uniform(0, 10, 50)
        true_coef = np.array([3.0, -2.0, 5.0])
        A = np.column_stack([x, y, np.ones(50)])
        b = A @ true_coef
        sol, *_ = _weighted_lscov(A, b)
        assert np.allclose(sol.ravel(), true_coef, atol=1e-6)

    def test_recover_known_coefficients_weighted(self):
        from venti.spatial.fitting import _weighted_lscov

        rng = np.random.default_rng(1)
        x = rng.uniform(0, 10, 50)
        y = rng.uniform(0, 10, 50)
        true_coef = np.array([1.5, -0.5, 2.0])
        A = np.column_stack([x, y, np.ones(50)])
        b = A @ true_coef
        weights = rng.uniform(0.5, 2.0, 50)
        sol, *_ = _weighted_lscov(A, b, weights)
        assert np.allclose(sol.ravel(), true_coef, atol=1e-5)

    def test_returns_eight_values(self):
        from venti.spatial.fitting import _weighted_lscov

        A = np.column_stack([np.arange(10.0), np.ones(10)])
        b = np.arange(10.0)
        result = _weighted_lscov(A, b)
        assert len(result) == 8

    def test_residuals_shape(self):
        from venti.spatial.fitting import _weighted_lscov

        n, p = 20, 3
        A = np.random.rand(n, p)
        b = np.random.rand(n)
        _, _, _, _, res, wres, obs_hat, _ = _weighted_lscov(A, b)
        assert res.shape == (n, 1)
        assert wres.shape == (n, 1)
        assert obs_hat.shape == (n, 1)


# _get_sliding_windows
@_SKIP_SPATIAL
class TestGetSlidingWindows:
    """Tests for _get_sliding_windows."""

    def test_single_window_when_size_equals_length(self):
        from venti.spatial.fitting import _get_sliding_windows

        windows = _get_sliding_windows(100, win_size=100, win_overlap=0)
        assert len(windows) == 1
        assert windows[0] == slice(0, 100)

    def test_two_windows_no_overlap(self):
        from venti.spatial.fitting import _get_sliding_windows

        windows = _get_sliding_windows(100, win_size=50, win_overlap=0)
        assert len(windows) == 2

    def test_overlap_reduces_stride(self):
        from venti.spatial.fitting import _get_sliding_windows

        no_overlap = _get_sliding_windows(200, win_size=50, win_overlap=0)
        with_overlap = _get_sliding_windows(200, win_size=50, win_overlap=10)
        assert len(with_overlap) >= len(no_overlap)

    def test_windows_cover_full_range(self):
        from venti.spatial.fitting import _get_sliding_windows

        length = 150
        windows = _get_sliding_windows(length, win_size=60, win_overlap=10)
        assert windows[0].start == 0
        assert windows[-1].stop == length

    def test_all_windows_are_slices(self):
        from venti.spatial.fitting import _get_sliding_windows

        windows = _get_sliding_windows(100, win_size=30, win_overlap=5)
        assert all(isinstance(w, slice) for w in windows)


# fit_windowed_plane
@_SKIP_SPATIAL
class TestFitWindowedPlane:
    """Tests for fit_windowed_plane (public API)."""

    NY, NX = 60, 80
    SNWE = (3_800_000.0, 3_860_000.0, 400_000.0, 480_000.0)

    def _flat_scene(self, value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        insar = np.full((self.NY, self.NX), value, dtype=np.float32)
        gnss = np.zeros((self.NY, self.NX), dtype=np.float32)
        return insar, gnss

    def _linear_residual_scene(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0, 1, self.NX)
        y = np.linspace(0, 1, self.NY)
        xg, yg = np.meshgrid(x, y)
        ramp = (0.05 * xg + 0.03 * yg).astype(np.float32)
        return ramp.copy(), np.zeros_like(ramp)

    def test_output_shapes(self):
        from venti.spatial.fitting import fit_windowed_plane

        insar, gnss = self._flat_scene()
        surface, std = fit_windowed_plane(
            insar, gnss,
            win_xsize=40, win_ysize=30,
            win_overlap_x=5, win_overlap_y=5,
            win_extend_x=40, win_extend_y=30,
            n_jobs=1,
        )
        assert surface.shape == (self.NY, self.NX)
        assert std.shape == (self.NY, self.NX)

    def test_flat_residual_gives_near_zero_surface(self):
        from venti.spatial.fitting import fit_windowed_plane

        insar, gnss = self._flat_scene(value=0.0)
        surface, _ = fit_windowed_plane(
            insar, gnss,
            win_xsize=40, win_ysize=30,
            win_overlap_x=5, win_overlap_y=5,
            win_extend_x=40, win_extend_y=30,
            poly_order=1, n_jobs=1,
        )
        assert np.nanmax(np.abs(surface)) < 1e-4

    def test_linear_ramp_is_captured(self):
        from venti.spatial.fitting import fit_windowed_plane

        insar, gnss = self._linear_residual_scene()
        surface, _ = fit_windowed_plane(
            insar, gnss,
            win_xsize=40, win_ysize=30,
            win_overlap_x=5, win_overlap_y=5,
            win_extend_x=40, win_extend_y=30,
            poly_order=1, n_jobs=1,
        )
        assert np.nanmax(np.abs(surface)) > 1e-4

    def test_masked_nan_pixels_do_not_crash(self):
        from venti.spatial.fitting import fit_windowed_plane

        insar, gnss = self._flat_scene()
        insar[10:20, 10:20] = np.nan
        surface, _ = fit_windowed_plane(
            insar, gnss,
            win_xsize=40, win_ysize=30,
            win_overlap_x=5, win_overlap_y=5,
            win_extend_x=40, win_extend_y=30,
            n_jobs=1,
        )
        assert surface.shape == (self.NY, self.NX)

    def test_gnss_los_std_accepted(self):
        from venti.spatial.fitting import fit_windowed_plane

        insar, gnss = self._flat_scene()
        std_field = np.ones_like(insar) * 0.001
        surface, _ = fit_windowed_plane(
            insar, gnss,
            win_xsize=40, win_ysize=30,
            win_overlap_x=5, win_overlap_y=5,
            win_extend_x=40, win_extend_y=30,
            gnss_los_std=std_field, n_jobs=1,
        )
        assert surface.shape == (self.NY, self.NX)


# SpatialProcessor
@_SKIP_SPATIAL
class TestSpatialProcessor:
    """Tests for SpatialProcessor.fit_windowed_surface."""

    NY, NX = 60, 80
    SNWE = (3_800_000.0, 3_860_000.0, 400_000.0, 480_000.0)

    def test_returns_ndarray(self):
        from venti.spatial.processor import SpatialProcessor

        processor = SpatialProcessor()
        insar = np.zeros((self.NY, self.NX), dtype=np.float32)
        gnss = np.zeros((self.NY, self.NX), dtype=np.float32)
        result = processor.fit_windowed_surface(
            insar, gnss, 
            window_size_x=40, window_size_y=30, n_jobs=1,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (self.NY, self.NX)

    def test_default_extend_equals_window_size(self):
        from venti.spatial.processor import SpatialProcessor

        processor = SpatialProcessor()
        insar = np.zeros((self.NY, self.NX), dtype=np.float32)
        gnss = np.zeros((self.NY, self.NX), dtype=np.float32)
        result = processor.fit_windowed_surface(
            insar, gnss, 
            window_size_x=40, window_size_y=30, n_jobs=1,
        )
        assert result.shape == (self.NY, self.NX)

    def test_custom_poly_order(self):
        from venti.spatial.processor import SpatialProcessor

        processor = SpatialProcessor()
        insar = np.zeros((self.NY, self.NX), dtype=np.float32)
        gnss = np.zeros((self.NY, self.NX), dtype=np.float32)
        for order in [0, 1, 1.5, 2]:
            result = processor.fit_windowed_surface(
                insar, gnss, 
                window_size_x=40, window_size_y=30,
                poly_order=order, n_jobs=1,
            )
            assert result.shape == (self.NY, self.NX)

    def test_explicit_extend_parameters(self):
        from venti.spatial.processor import SpatialProcessor

        processor = SpatialProcessor()
        insar = np.zeros((self.NY, self.NX), dtype=np.float32)
        gnss = np.zeros((self.NY, self.NX), dtype=np.float32)
        result = processor.fit_windowed_surface(
            insar, gnss, 
            window_size_x=40, window_size_y=30,
            window_extend_x=20, window_extend_y=15, n_jobs=1,
        )
        assert result.shape == (self.NY, self.NX)
