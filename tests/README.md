# Venti Unit Tests

## Overview

This directory contains unit tests for the Venti package.

## Running Tests

### Using pytest

From your conda environment:

```bash
# Activate your environment
conda activate nlab

# Run all tests
pytest

# Run specific test file
pytest tests/test_unwrap_corrections.py -v

# Run basic smoke tests
pytest tests/test_basic.py -v
```

### Test Files

- **test_basic.py** - Simple smoke tests for core functionality
- **test_unwrap_corrections.py** - Comprehensive tests for the unwrap corrections module

## Test Coverage

### test_unwrap_corrections.py

Tests for the unwrap error correction module:

1. **TestUnwrapCorrector** - Tests for the main class
   - Initialization (default and custom parameters)
   - Displacement preparation
   - Watershed segmentation
   - Regional median computation
   - Unwrap cycle computation (including NaN handling)
   - End-to-end correction

2. **TestReadNetCDF** - Tests for NetCDF reading (requires rasterio)
   - Basic NetCDF structure
   - GeoTransform attribute handling (OPERA format)
   - Error handling for missing variables

3. **TestCorrectRegionOffset** - Tests for convenience function
   - Array inputs
   - NetCDF file inputs
   - GeoTIFF output (requires rasterio)

4. **TestSaveGeoTIFF** - Tests for GeoTIFF export (requires rasterio)
   - Explicit transform and CRS
   - Error handling

### test_basic.py

Simple smoke tests that verify:
- Core imports work without triggering pyproj
- Basic initialization
- Simple corrections work

## Dependencies

### Required
- numpy
- pytest
- scikit-image
- scipy
- numba
- xarray

### Optional
- rasterio (for GeoTIFF I/O tests)
- pytest-cov (for coverage reports)
- pytest-randomly (for randomized test order)

Tests that require optional dependencies will be automatically skipped if the dependency is not installed.

## Known Issues

### Numpy Binary Incompatibility Warnings

If you see warnings like:
```
RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility
```

This is a conda environment issue between xarray/netCDF4 and numpy versions, not a problem with the Venti code. These warnings are filtered in the test files.

### PyProj Initialization

The main venti package uses lazy loading to avoid pyproj initialization errors when only using the unwrap module. If you encounter pyproj errors, ensure you're using the unwrap module directly:

```python
from venti.unwrap import UnwrapCorrector  # Good - lazy loads
import venti; venti.models  # May trigger pyproj errors
```
