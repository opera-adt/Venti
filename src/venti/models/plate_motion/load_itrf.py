import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import os

# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent

def load_itrf14_json(filepath: Optional[Union[str, Path]] = None) -> Dict:
    """
    Load ITRF2024 data from JSON format.
    https://academic.oup.com/gji/article/209/3/1906/3095992#supplementary-data
    
    Args:
        filepath: Path to JSON file. If None, uses default itrf2024.json
        
    Returns:
        Dictionary with metadata and plates data
    """
    if filepath is None:
        filepath = MODULE_DIR / "data/itrf2014.json"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

def json_to_dataframe(json_data: Dict) -> pd.DataFrame:
    """
    Convert JSON format ITRF data to pandas DataFrame.
    
    Args:
        json_data: Dictionary from load_itrf_json()
        
    Returns:
        DataFrame with columns: plate, name, omega_x, omega_y, omega_z
    """
    rows = []
    for plate_code, plate_data in json_data['plates'].items():
        rows.append({
            'plate': plate_code,
            'name': plate_data['name'],
            'omega_x': plate_data['omega_x'],
            'omega_y': plate_data['omega_y'],
            'omega_z': plate_data['omega_z']
        })
    
    return pd.DataFrame(rows)

def convert_to_euler_poles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ITRF rotation rates to Euler pole representation.
    
    Args:
        df: DataFrame with omega_x, omega_y, omega_z columns (in deg/Myr)
        
    Returns:
        DataFrame with Euler pole parameters (lon, lat, omega)
    """
    from .euler_pole import rotation_rate_to_euler_pole
    
    results = []
    
    for _, row in df.iterrows():
        # Convert from deg/Myr to rad/year
        wx = np.radians(row['omega_x']) / 1e6  # deg/Myr → rad/year
        wy = np.radians(row['omega_y']) / 1e6
        wz = np.radians(row['omega_z']) / 1e6
        
        # Convert to Euler pole
        euler_lon, euler_lat, omega = rotation_rate_to_euler_pole(wx, wy, wz)
        
        result = {
            'plate': row['plate'],
            'euler_longitude': euler_lon,
            'euler_latitude': euler_lat,
            'angular_velocity': omega  # deg/Myr
        }
        
        # Include plate name if available
        if 'name' in row:
            result['name'] = row['name']
            
        results.append(result)
    
    return pd.DataFrame(results)


def get_plate_data(plate_code: str) -> Dict:
    """
    Get data for a specific plate.
    
    Args:
        plate_code: Plate abbreviation (e.g., 'PCFC', 'NOAM')
        format_type: 'csv' or 'json'
        
    Returns:
        Dictionary with plate rotation parameters
    """
    json_data = load_itrf14_json()
    df = json_to_dataframe(json_data)
    
    plate_row = df[df['plate'] == plate_code.upper()]
    if len(plate_row) == 0:
        available_plates = df['plate'].tolist()
        raise ValueError(f"Plate '{plate_code}' not found. Available: {available_plates}")
    
    return plate_row.iloc[0].to_dict()


