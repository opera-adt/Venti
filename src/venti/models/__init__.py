from .plate_motion import (
    # Main calculation functions
    calculate_euler_pole,
    get_euler_pole_uncertainty,

    # Prediction functions
    model_plate_velocities,
    model_velocities_from_euler_pole, 

    # Conversion utilities
    rotation_rate_to_euler_pole,
    euler_pole_to_rotation_rate,
    
    # Coordinate system functions
    get_conversion_matrix,
    get_local_frame,
    
    
    # Constants
    EARTH_RADIUS_KM,
    DEG_TO_RAD,
    RAD_TO_DEG,
    MYR_TO_YEAR
)

# Import ITRF data loading utilities
from .load_itrf import (
    ITRF14_DATA,
    ITRF20_DATA,
    load_itrf_pmm,
    json_to_dataframe,
    convert_to_euler_poles,
    get_plate_data)

from .load_gia import(
    ICE6D_URL,
    CARON_GIA,
    download_ice6g_data,
    load_ice6g_model,
    load_caron_model,
    clip_gia_df,
    rasterize_gdf)

# Package metadata
__all__ = [
    # Euler pole calculation functions
    'calculate_euler_pole',
    'get_euler_pole_uncertainty', 
    'rotation_rate_to_euler_pole',
    'euler_pole_to_rotation_rate',
    'get_conversion_matrix',
    'get_local_frame',
    
    # Modeling functions
    'model_plate_velocities',
    'model_velocities_from_euler_pole',
    
    # ITRF data loading functions
    'load_itrf_pmm',
    'json_to_dataframe',
    'convert_to_euler_poles',
    'get_plate_data',

    # GIA data loading functions
    'download_ice6g_data',
    'load_ice6g_model',
    'load_caron_model', 
    
    # GIA clipping functions
    'clip_gia_df',
    'rasterize_gdf'

    # Constants
    'EARTH_RADIUS_KM',
    'DEG_TO_RAD', 
    'RAD_TO_DEG',
    'MYR_TO_YEAR',
    'ITRF14_DATA',
    'ITRF20_DATA',
    'ICE6D_URL',
    'CARON_GIA'
]