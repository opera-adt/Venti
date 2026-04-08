# Venti

**Integrate GNSS calibration with InSAR products for accurate vertical land motion estimation**

---

## Overview

Venti is a Python toolkit designed to fuse GNSS data with OPERA-DISP product to calibrate and project InSAR line-of-sight measurements into vertical displacement estimates.

---

## Features

- GNSS calibration of InSAR products
- Projection from line-of-sight (LOS) to vertical displacement
- Line-of-sight (LOS) Decomposition

---

## Installation

```bash
git clone https://github.com/opera-adt/Venti.git
cd Venti
pip install -r requirements.txt

1. Download source code:
```bash
git clone https://github.com/opera-adt/Venti.git
```
2. Install dependencies, either to a new environment:
```bash
mamba env create --name venti-env --file Venti/environment.yml
conda activate venti-env
```
or install within your existing env with mamba.

3. Install `venti` via pip in editable mode
```bash
python -m pip install .
```
or in editable mode:
```bash
python -m pip install -e .
```

---

## Data Staging

Before running any calibration or decomposition workflow, you need to download and
prepare all input data for the target frame. The staging workflow handles this end-to-end.

### What gets staged

For a given OPERA frame ID and secondary date, staging prepares:

| Step | Output | Location |
|------|--------|----------|
| 1. Download DISP-S1 product | OPERA NetCDF interferogram | `<output_dir>/disp_s1/` |
| 2. Generate DEM | GLO30 GeoTIFF in WGS84 and native UTM | `<output_dir>/dem/` |
| 3. Generate LOS geometry | LOS ENU raster + incidence angle | `<output_dir>/los/` |
| 4. Tropospheric corrections | HRRR-based correction GeoTIFFs | `<output_dir>/tropo/` |
| 5. UNR GNSS velocities | Per-station `.tenv8` files + `velocities.parquet` | `<output_dir>/gnss/` |

DEM and LOS outputs are idempotent — re-running staging for a new date on the same
frame reuses existing files.

### CLI

```bash
cd scripts/staging

# Stage all data for a single frame and date
python stage_frame_cli.py --frame-id 8887 --date 2016-06-15

# Custom output directory
python stage_frame_cli.py --frame-id 8887 --date 2016-06-15 \
    --output-dir /data/opera/frame_8887

# Skip optional steps
python stage_frame_cli.py --frame-id 8887 --date 2016-06-15 \
    --skip-tropo --skip-gnss

# Full options
python stage_frame_cli.py --help
```

All CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--frame-id` | required | OPERA frame identifier |
| `--date` | required | Secondary date (YYYY-MM-DD or YYYYMMDD) |
| `--output-dir` | `./staging` | Root directory for all outputs |
| `--num-workers` | `4` | Parallel workers for downloads |
| `--dem-buffer` | `10000.0` | Buffer in meters around frame for DEM |
| `--skip-tropo` | `False` | Skip tropospheric corrections |
| `--skip-gnss` | `False` | Skip GNSS download and velocity estimation |
| `--gnss-reference-frame` | `IGS20` | GNSS reference frame (`IGS20` or `IGS14`) |
| `--gnss-padding` | `0.0` | Extra meters beyond frame bounds for station search |
| `--gnss-start-year` | `2014.0` | Earliest year used for velocity estimation |

### Python API

```python
from pathlib import Path
from venti.workflow import run_data_staging

# Stage all data for a single interferogram
run_data_staging(
    frame_id=8887,
    date="2016-06-15",
    output_dir=Path("./data"),
)
```

Skip individual steps as needed:

```python
run_data_staging(
    frame_id=8887,
    date="2016-06-15",
    output_dir=Path("./data"),
    skip_tropo=True,
    skip_gnss=True,
)
```

### Preview available products before staging

Use `disp_cli.py` to check what products exist for a frame before downloading:

```bash
cd scripts/staging
python disp_cli.py preview --frame-id 8887 --start 2016-01-01 --end 2017-01-01 --print-dates
```

### Output directory structure

```
<output_dir>/
├── disp_s1/
│   └── OPERA_L3_DISP-S1_IW_F08887_VV_*.nc
├── dem/
│   ├── dem_frame_8887.tif               # WGS84
│   └── dem_frame_8887_epsg32610.tif     # native UTM
├── los/
│   ├── los_enu_frame_8887.tif           # 3-band ENU raster
│   ├── incidence_angle_frame_8887.tif
│   ├── los_east.vrt
│   ├── los_north.vrt
│   └── los_up.vrt
├── tropo/
│   ├── tropo_urls.txt
│   ├── cropped_tropo/
│   ├── tropo_corrections/
│   └── tropo_corrections_{epsg}/
└── gnss/
    ├── grid_latlon_lookup.txt
    ├── stations/
    │   └── *.tenv8
    └── velocities.parquet
```

---

## Running from CLI

### 1. Generate configuration templates

```bash
python -m venti config --output-dir configs/
```

This writes two files to `configs/`:
- `runconfig.yaml` — run-specific settings (input/output paths, workflow type)
- `algorithm_parameters.yaml` — algorithm defaults that rarely need changing

### 2. Edit `runconfig.yaml`

Fill in the section that matches your workflow. For calibration:

```yaml
calibration_input_group:
  input_files: path/to/displacement/files   # directory of OPERA DISP NetCDF files
  los_file:    path/to/los_vectors.tif      # 3-band GeoTIFF (east, north, up)
  water_mask:  path/to/water_mask.tif       # GeoTIFF (1=land, 0=water)

product_path_group:
  product_path: output/

primary_executable:
  workflow_name: calibrate                  # or 'decompose'
```

### 3. Run the workflow

Two modes are available depending on whether you want to process a full directory of files or a single epoch.

**Batch run** — calibrates all `.nc` files found in `input_files`:

```bash
python -m venti run configs/runconfig.yaml
```

**Single-file run** — calibrates one specified displacement file:

```bash
python -m venti run-single configs/runconfig.yaml /path/to/epoch_001.nc
```

With an optional tropospheric correction for that epoch:

```bash
python -m venti run-single configs/runconfig.yaml /path/to/epoch_001.nc \
    --tropo-file /path/to/tropo_001.tif
```

`algorithm_parameters.yaml` is auto-discovered from the same directory as `runconfig.yaml`, so keep both files together. Increase verbosity with `--log-level DEBUG` on either command.

---

## Running from Python

**Batch run:**

```python
from venti.workflow.calibration import CalibrationWorkflow
from venti.workflow.config import load_config

config = load_config("configs/runconfig.yaml")
workflow = CalibrationWorkflow(config=config)
state = workflow.run()

print(f"Processed: {state.n_files_processed} / {state.n_files_total}")
for f in state.output_files:
    print(f"  {f.name}")
```

**Single-file run:**

```python
from pathlib import Path

from venti.workflow.calibration import CalibrationWorkflow
from venti.workflow.config import load_config

config = load_config("configs/runconfig.yaml")
workflow = CalibrationWorkflow(config=config)
state = workflow.run_single(disp_file=Path("/path/to/epoch_001.nc"))

print(f"Output: {state.output_files[0]}")
```

`load_config` auto-discovers `algorithm_parameters.yaml` from the same directory as `runconfig.yaml`. You can also build the configuration fully in Python — see the [calibration workflow notebook](notebooks/calibration_workflow.ipynb) for a complete example.

---

### Setup for contributing


We use [pre-commit](https://pre-commit.com/) to automatically run linting, formatting, and [mypy type checking](https://www.mypy-lang.org/).
Additionally, we follow [`numpydoc` conventions for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
To install pre-commit locally, run:

```bash
pre-commit install
```
This adds a pre-commit hooks so that linting/formatting is done automatically. If code does not pass the checks, you will be prompted to fix it before committing.
Remember to re-add any files you want to commit which have been altered by `pre-commit`. You can do this by re-running `git add` on the files.

Since we use [black](https://black.readthedocs.io/en/stable/) for formatting and [flake8](https://flake8.pycqa.org/en/latest/) for linting, it can be helpful to install these plugins into your editor so that code gets formatted and linted as you save.

### Running the unit tests

After making functional changes and/or have added new tests, you should run pytest to check that everything is working as expected.

First, install the extra test dependencies:
```bash
python -m pip install --no-deps -e .[test]
```

Then run the tests:

```bash
pytest
```
