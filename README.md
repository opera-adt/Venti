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
