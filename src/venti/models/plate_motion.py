# SPDX-FileCopyrightText: 2025-2026 opera-adt
# SPDX-License-Identifier: BSD-3-Clause
# Part of Venti, https://github.com/opera-adt/Venti (BSD-3-Clause, see LICENSE).
# Adapted from geepers (https://github.com/opera-adt/geepers, Apache-2.0, see
# its NOTICE).
"""Rigid plate motion from the published ITRF plate-motion models.

The ``plate_motion`` layer of DISP-CAL and the ``plate`` of each frame in the
frame-parameter table. The pole maths is `geepers.euler`; the ITRF2020/2014-PMM
rotation vectors and the UNR plate codes are Venti's own (``models/data``),
because opera-adt/geepers does not ship them.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from geepers.euler import (
    EulerPole,
    predict_plate_motion,
    rotation_vector_to_pole,
)
from numpy.typing import ArrayLike

__all__ = [
    "PLATE_CODES",
    "EulerPole",
    "load_plate_motion_model",
    "plate_pole",
    "plate_velocity_enu",
]

_DATA_DIR = Path(__file__).parent / "data"
PlateModel = Literal["ITRF2020-PMM", "ITRF2014-PMM"]
PLATE_MODEL_FILES: dict[str, str] = {
    # Altamimi et al. 2023, doi:10.1029/2023GL106373
    "ITRF2020-PMM": "itrf2020_pmm.json",
    # Altamimi et al. 2017, doi:10.1093/gji/ggx136
    "ITRF2014-PMM": "itrf2014_pmm.json",
}
#: UNR two-letter plate codes (the frames of the UNR grid products) -> ITRF PMM
#: plate names.
PLATE_CODES: dict[str, str] = {
    "NA": "NOAM",
    "PA": "PCFC",
    "CA": "CARB",
    "EU": "EURA",
    "AU": "AUST",
    "AN": "ANTA",
    "AR": "ARAB",
    "IN": "INDI",
    "NZ": "NAZC",
    "SA": "SOAM",
    "SO": "SOMA",
    "AF": "NUBI",
    "AM": "AMUR",
}


@cache
def load_plate_motion_model(model: PlateModel = "ITRF2020-PMM") -> dict[str, dict]:
    """Load a published plate-motion model: plate name -> rotation components.

    Returns
    -------
    dict
        ``{"NOAM": {"name": "North American Plate", "omega_x": ..., "omega_y":
        ..., "omega_z": ...}, ...}`` with the rotation-vector components in
        degrees per million years (geocentric Cartesian, ITRF frame).

    """
    try:
        filename = PLATE_MODEL_FILES[model]
    except KeyError as e:
        msg = (
            f"Unknown plate-motion model {model!r}; "
            f"choose from {sorted(PLATE_MODEL_FILES)}"
        )
        raise ValueError(msg) from e
    with (_DATA_DIR / filename).open() as f:
        return json.load(f)["plates"]


def plate_pole(plate: str, model: PlateModel = "ITRF2020-PMM") -> EulerPole:
    """Euler pole of a tectonic plate from a published plate-motion model.

    Parameters
    ----------
    plate : str
        PMM plate name (``"NOAM"``) or UNR two-letter code (``"NA"``); see
        `PLATE_CODES`.
    model : {"ITRF2020-PMM", "ITRF2014-PMM"}
        Which model. Default ITRF2020-PMM, the frame of the UNR IGS20 grid.

    Returns
    -------
    EulerPole
        Pole position (deg) and rate (deg/Myr); no covariance is published
        in the tables, so the uncertainty fields are NaN.

    """
    plates = load_plate_motion_model(model)
    name = PLATE_CODES.get(plate.upper(), plate.upper())
    if name not in plates:
        msg = (
            f"Plate {plate!r} is not in {model}; names: {sorted(plates)}, "
            f"codes: {sorted(PLATE_CODES)}"
        )
        raise ValueError(msg)
    omega = plates[name]
    w = np.radians([omega["omega_x"], omega["omega_y"], omega["omega_z"]]) * 1e-6
    lon, lat, rate = rotation_vector_to_pole(w)
    return EulerPole(lon=lon, lat=lat, rate=rate)


def plate_velocity_enu(
    lon: ArrayLike,
    lat: ArrayLike,
    plate: str,
    model: PlateModel = "ITRF2020-PMM",
    height: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid-plate ENU velocity (mm/yr) of `plate` at the given points.

    A rigid rotation about the geocentre has no vertical component, so the
    up component is zero. This is the ``plate_motion`` layer of DISP-CAL:
    subtract it (projected to LOS and scaled by the pair's time span) from an
    IGS20-calibrated product to obtain a plate-fixed one.

    Parameters
    ----------
    lon, lat : array-like
        Points in degrees.
    plate : str
        PMM plate name or UNR two-letter code (``"NA"``, ``"PA"``, ``"CA"``).
    model : {"ITRF2020-PMM", "ITRF2014-PMM"}
        Plate-motion model.
    height : array-like, optional
        Ellipsoidal heights in meters.

    Returns
    -------
    ve, vn, vu : np.ndarray
        East, north and up velocities in mm/yr (``vu`` is all zeros).

    """
    ve, vn = predict_plate_motion(plate_pole(plate, model), lon, lat, height)
    return ve, vn, np.zeros_like(ve)
