# SPDX-FileCopyrightText: 2025-2026 opera-adt
# SPDX-License-Identifier: BSD-3-Clause
# Part of Venti, https://github.com/opera-adt/Venti (BSD-3-Clause, see LICENSE).
"""Plate motion: geepers' Euler maths on Venti's ITRF PMM tables (ADR-0022)."""

from __future__ import annotations

import numpy as np
import pytest

import venti.models as models
from venti.models import plate_motion


def test_models_exports_plate_motion():
    assert models.plate_velocity_enu is plate_motion.plate_velocity_enu
    assert models.plate_pole is plate_motion.plate_pole
    assert models.PLATE_CODES["NA"] == "NOAM"


def test_pole_maths_comes_from_geepers():
    from geepers import euler

    assert plate_motion.EulerPole is euler.EulerPole


@pytest.mark.parametrize(
    ("model", "lon", "lat", "rate"),
    [
        # Altamimi et al. 2023 (ITRF2020-PMM) and 2017 (ITRF2014-PMM), NOAM
        ("ITRF2020-PMM", -86.1, -8.35, 0.187),
        ("ITRF2014-PMM", -88.0, -5.2, 0.194),
    ],
)
def test_north_america_pole_matches_the_published_model(model, lon, lat, rate):
    pole = plate_motion.plate_pole("NA", model)
    assert pole.lon == pytest.approx(lon, abs=0.5)
    assert pole.lat == pytest.approx(lat, abs=0.5)
    assert pole.rate == pytest.approx(rate, abs=0.005)


def test_codes_and_names_give_the_same_pole():
    by_code = plate_motion.plate_pole("pa")
    by_name = plate_motion.plate_pole("PCFC")
    assert (by_code.lon, by_code.lat, by_code.rate) == (
        by_name.lon,
        by_name.lat,
        by_name.rate,
    )


def test_every_code_is_in_the_itrf2020_model():
    plates = plate_motion.load_plate_motion_model()
    assert set(plate_motion.PLATE_CODES.values()) <= set(plates)


def test_unknown_plate_and_model_are_rejected():
    with pytest.raises(ValueError, match="not in ITRF2020-PMM"):
        plate_motion.plate_pole("XX")
    with pytest.raises(ValueError, match="Unknown plate-motion model"):
        plate_motion.load_plate_motion_model("ITRF2008-PMM")  # type: ignore[arg-type]


def test_rigid_plate_velocity_has_no_up_component():
    ve, vn, vu = plate_motion.plate_velocity_enu([-95.4], [29.8], "NA")
    # Houston in ITRF2020: ~12-13 mm/yr west, a few mm/yr south
    assert -16 < ve[0] < -9
    assert vn[0] < 0
    assert vu[0] == 0


def test_get_model_rates_uses_venti_plate_motion():
    """The PMM raster helper produces mm/yr grids from Venti's pole."""
    gpd = pytest.importorskip("geopandas")
    import importlib.util
    from pathlib import Path

    from shapely.geometry import box

    spec = importlib.util.spec_from_file_location(
        "get_model_rates",
        Path(__file__).resolve().parents[1] / "scripts/get_model_rates.py",
    )
    gmr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gmr)
    frame = gpd.GeoDataFrame(geometry=[box(-96.0, 29.0, -95.0, 30.0)], crs="EPSG:4326")
    ve, vn, attrs = gmr.get_frame_pmm(
        frame,
        plate="NA",
        date=2020,
        grid_posting=10000,  # frame in EPSG:4326, posting in metres
    )
    assert ve.shape == vn.shape == (attrs["height"], attrs["width"])
    assert attrs["units"] == "mm/year"
    assert attrs["model"] == "ITRF2020"
    assert np.all((ve > -16) & (ve < -9))
    assert np.all(vn < 0)
