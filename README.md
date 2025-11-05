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
editable mode
```bash
python -m pip install --n
```

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
