"""Basic smoke tests that don't depend on other venti modules."""

import numpy as np
import pytest


def test_numpy_available():
    """Test that numpy is available."""
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6


def test_can_import_unwrap():
    """Test that we can import the unwrap module without triggering pyproj."""
    # This should not trigger models import
    from venti.unwrap import UnwrapCorrector

    assert UnwrapCorrector is not None


def test_unwrap_corrector_init():
    """Test UnwrapCorrector initialization."""
    from venti.unwrap import UnwrapCorrector

    corrector = UnwrapCorrector()
    assert corrector.wavelength == 0.0555
    assert corrector.min_region_area == 20


def test_unwrap_corrector_custom_init():
    """Test UnwrapCorrector with custom parameters."""
    from venti.unwrap import UnwrapCorrector

    corrector = UnwrapCorrector(min_region_area=50, wavelength=0.06)
    assert corrector.wavelength == 0.06
    assert corrector.min_region_area == 50


def test_simple_correction():
    """Test basic correction without needing actual data files."""
    from venti.unwrap import UnwrapCorrector

    corrector = UnwrapCorrector(wavelength=0.1, min_region_area=5)

    # Create simple synthetic data
    disp = np.zeros((30, 30))
    disp[5:15, 5:15] = 0.5
    disp[15:25, 15:25] = 0.6

    mask = np.ones((30, 30), dtype=bool)

    corrected = corrector.correct(disp, mask)

    assert corrected is not None
    assert corrected.shape == disp.shape
    assert isinstance(corrected, np.ma.MaskedArray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
