import json
from pathlib import Path

import numpy as np
import pandas as pd

# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent
ITRF14_DATA = MODULE_DIR / "data/itrf2014_sites.csv"
ITRF20_DATA = MODULE_DIR / "data/itrf2020_sites.csv"


def json_to_dataframe(json_data: dict) -> pd.DataFrame:
    """Convert JSON format ITRF data to pandas DataFrame.

    Args:
        json_data: Dictionary from load_itrf_json()

    Returns:
        DataFrame with columns: plate, name, omega_x, omega_y, omega_z

    """
    rows = []
    for plate_code, plate_data in json_data["plates"].items():
        rows.append(
            {
                "plate": plate_code,
                "name": plate_data["name"],
                "omega_x": plate_data["omega_x"],
                "omega_y": plate_data["omega_y"],
                "omega_z": plate_data["omega_z"],
            }
        )

    return pd.DataFrame(rows)


def load_itrf_pmm(
    filepath: str | Path | None = None, date: int = 2020
) -> dict:
    """Load ITRF Plate Motion Model from JSON format.

    Args:
        filepath: Path to JSON file. If None, uses default itrf{date}.json
        date: ITRF reference frame year (e.g., 2014, 2020)

    Returns:
        Dictionary with metadata and plates data

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the JSON file is invalid or malformed

    """
    if filepath is None:
        filepath = MODULE_DIR / f"data/itrf{date}_pmm.json"

    try:
        with open(filepath) as f:
            data = json.load(f)

        # Basic validation
        if "metadata" not in data or "plates" not in data:
            msg = f"Invalid ITRF JSON format in {filepath}"
            raise ValueError(msg)

        return json_to_dataframe(data)

    except FileNotFoundError:
        msg = f"ITRF data file not found: {filepath}"
        raise FileNotFoundError(msg)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {filepath}: {e}"
        raise ValueError(msg)


def convert_to_euler_poles(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ITRF rotation rates to Euler pole representation.

    Args:
        df: DataFrame with omega_x, omega_y, omega_z columns (in deg/Myr)

    Returns:
        DataFrame with Euler pole parameters (lon, lat, omega)

    """
    from .plate_motion.euler_pole import rotation_rate_to_euler_pole

    results = []

    for _, row in df.iterrows():
        # Convert from deg/Myr to rad/year
        wx = np.radians(row["omega_x"]) / 1e6  # deg/Myr → rad/year
        wy = np.radians(row["omega_y"]) / 1e6
        wz = np.radians(row["omega_z"]) / 1e6

        # Convert to Euler pole
        euler_lon, euler_lat, omega = rotation_rate_to_euler_pole(wx, wy, wz)

        result = {
            "plate": row["plate"],
            "euler_longitude": euler_lon,
            "euler_latitude": euler_lat,
            "angular_velocity": omega,  # deg/Myr
        }

        # Include plate name if available
        if "name" in row:
            result["name"] = row["name"]

        results.append(result)

    return pd.DataFrame(results)


def get_plate_data(plate_code: str, date: int = 2020) -> dict:
    """Get data for a specific plate.

    Args:
        plate_code: Plate abbreviation (e.g., 'PCFC', 'NOAM')
        format_type: 'csv' or 'json'

    Returns:
        Dictionary with plate rotation parameters

    """
    df = load_itrf_pmm(date=date)

    plate_row = df[df["plate"] == plate_code.upper()]
    if len(plate_row) == 0:
        available_plates = df["plate"].tolist()
        msg = f"Plate '{plate_code}' not found. Available: {available_plates}"
        raise ValueError(
            msg
        )

    return plate_row.iloc[0].to_dict()
