"""Euler Pole Calculation for Tectonic Plate Motion.

This module provides functions to calculate Euler poles from GPS velocity data,
handling coordinate transformations and uncertainty estimation.
"""


import numpy as np
from pyproj import Transformer

# Constants
EARTH_RADIUS_KM = 6371.0
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
MYR_TO_YEAR = 1e6
MAS_TO_RAD = np.pi / 648000000


# Initialize coordinate transformer (WGS84 to geocentric cartesian)
LONLAT_TO_XYZ_TRANSFORMER = Transformer.from_crs(
    "EPSG:4326",
    {"proj": "geocent", "ellps": "GRS80", "datum": "WGS84"},
    always_xy=True,
)


def get_conversion_matrix(longitude: float, latitude: float) -> np.ndarray:
    """Calculate the conversion matrix from XYZ to ENU (East-North-Up) coordinates.

    Args:
        longitude: Longitude in degrees
        latitude: Latitude in degrees

    Returns:
        3x3 conversion matrix R

    """
    lon_rad = np.radians(longitude)
    lat_rad = np.radians(latitude)

    cos_lon, sin_lon = np.cos(lon_rad), np.sin(lon_rad)
    cos_lat, sin_lat = np.cos(lat_rad), np.sin(lat_rad)

    R = np.array(
        [
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )

    return R


def get_local_frame(
    longitude: float, latitude: float, height: float = 0.0
) -> np.ndarray:
    """Calculate the local reference frame matrix for a given position.

    Args:
        longitude: Longitude in degrees
        latitude: Latitude in degrees
        height: Height above ellipsoid in meters (default: 0)

    Returns:
        3x3 local frame matrix RAi

    """
    # Convert to geocentric cartesian coordinates
    x, y, z = LONLAT_TO_XYZ_TRANSFORMER.transform(
        longitude, latitude, height, radians=False
    )

    # Get conversion matrix XYZ to ENU
    R = get_conversion_matrix(longitude, latitude)

    # Observation equation matrix (skew-symmetric for cross product)
    # ω × r = [ωy*z - ωz*y, ωz*x - ωx*z, ωx*y - ωy*x]
    Ai = np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]], dtype=np.float64)

    # Apply rotation matrix
    RAi = np.dot(R, Ai)

    return RAi


def rotation_rate_to_euler_pole(
    wx: float, wy: float, wz: float
) -> tuple[float, float, float]:
    """Convert rotation rate vector to Euler pole parameters.

    Args:
        wx, wy, wz: Rotation rate components in rad/year

    Returns:
        Tuple of (euler_longitude, euler_latitude, omega) in degrees and deg/Myr

    """
    W_magnitude = np.sqrt(wx**2 + wy**2 + wz**2)

    # Angular velocity in deg/Myr from rad/year
    # rad/year × (180°/π rad) × (1e6 year/Myr) = deg/Myr
    omega = np.degrees(W_magnitude) * MYR_TO_YEAR

    # Euler pole latitude (colatitude from z-axis)
    euler_latitude = 90.0 - np.degrees(np.arccos(wz / W_magnitude))

    # Euler pole longitude
    euler_longitude = np.degrees(np.arctan2(wy, wx))

    return euler_longitude, euler_latitude, omega


def euler_pole_to_rotation_rate(
    euler_longitude: float, euler_latitude: float, omega: float
) -> tuple[float, float, float]:
    """Convert Euler pole parameters to rotation rate vector.

    Args:
        euler_longitude: Euler pole longitude in degrees
        euler_latitude: Euler pole latitude in degrees
        omega: Angular velocity in deg/Myr

    Returns:
        Tuple of (wx, wy, wz) rotation rates in rad/year

    """
    # Convert angular velocity from deg/Myr to rad/year
    # deg/Myr × (π rad/180°) × (1 Myr/1e6 year) = rad/year
    omega_rad_per_year = np.radians(omega) / MYR_TO_YEAR

    cos_lon = np.cos(np.radians(euler_longitude))
    sin_lon = np.sin(np.radians(euler_longitude))
    cos_lat = np.cos(np.radians(euler_latitude))
    sin_lat = np.sin(np.radians(euler_latitude))

    wx = cos_lat * cos_lon * omega_rad_per_year
    wy = cos_lat * sin_lon * omega_rad_per_year
    wz = sin_lat * omega_rad_per_year

    return wx, wy, wz


def calculate_euler_pole(
    longitude: np.ndarray,
    latitude: np.ndarray,
    velocity_east: np.ndarray,
    velocity_north: np.ndarray,
    sigma_east: np.ndarray,
    sigma_north: np.ndarray,
    heights: np.ndarray | None = None,
    correlation_coefficient: float | None = 0,
) -> tuple[float, float, float, dict]:
    """Calculate Euler pole from GPS velocity data using weighted least squares.

    Args:
        longitude: Site longitudes in degrees
        latitude: Site latitudes in degrees
        velocity_east: East velocity components in mm/year
        velocity_north: North velocity components in mm/year
        sigma_east: East velocity uncertainties in mm/year
        sigma_north: North velocity uncertainties in mm/year
        heights: Site heights above ellipsoid in meters (optional)

    Returns:
        Tuple of (euler_longitude, euler_latitude, omega, statistics)
        where statistics is a dict containing fit quality metrics

    """
    n_sites = len(longitude)

    if heights is None:
        heights = np.zeros(n_sites)

    # Initialize design matrix and observation vector
    A = np.zeros((2 * n_sites, 3), dtype=np.float64)
    b = np.zeros((2 * n_sites, 1), dtype=np.float64)
    cov = np.zeros((2 * n_sites, 2 * n_sites), dtype=np.float64)

    # Build system matrices
    for i, (lon, lat, hgt, ve, vn, se, sn) in enumerate(
        zip(
            longitude,
            latitude,
            heights,
            velocity_east,
            velocity_north,
            sigma_east,
            sigma_north, strict=False,
        )
    ):
        # Get local frame matrix
        local_frame = get_local_frame(lon, lat, hgt)

        # Correlation coefficient should be dimensionless between -1 and 1
        # Covariance = σe × σn × correlation_coefficient
        cross_correlation = se * sn * correlation_coefficient

        # Fill design matrix
        A[2 * i, :] = local_frame[0, :]  # East component
        A[2 * i + 1, :] = local_frame[1, :]  # North component

        # Fill observation vector
        b[2 * i, 0] = ve
        b[2 * i + 1, 0] = vn

        # Fill covariance matrix
        cov[2 * i, 2 * i] = se**2
        cov[2 * i + 1, 2 * i + 1] = sn**2
        cov[2 * i + 1, 2 * i] = cross_correlation
        cov[2 * i, 2 * i + 1] = cross_correlation

    # Weighted least squares solution
    eigenvals = np.linalg.eigvals(cov)
    if np.any(eigenvals <= 0):
        msg = "Covariance matrix is not positive definite"
        raise ValueError(msg)

    # Solve linear system

    try:
        P = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        msg = "Covariance matrix is singular."
        raise ValueError(msg)

    ATP = A.T @ P

    N = ATP @ A
    M = ATP @ b

    # Solve for rotation parameters
    try:
        Q = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        msg = "Normal matrix is singular."
        raise ValueError(msg)

    X = np.dot(Q, M)

    # Calculate model predictions and residuals
    MP = np.dot(A, X)
    residuals = b - MP

    # Statistical analysis
    chi2 = float(residuals.T @ P @ residuals)
    # Deegres of freedom
    dof = 2 * n_sites - 3
    if dof <= 0:
        msg = f"Insufficient degrees of freedom: {dof}. Need at least 3 sites."
        raise ValueError(
            msg
        )

    reduced_chi2 = np.sqrt(chi2 / dof)

    # Extract rotation components
    # mm/year per km × (1 m/1000 mm) × (1 km/1000 m) = 1e-6 × (unitless) = rad/year
    wx = X[0, 0] / 1000  # rad/year
    wy = X[1, 0] / 1000  # rad/year
    wz = X[2, 0] / 1000  # rad/year

    # Convert to Euler pole
    euler_longitude, euler_latitude, omega = rotation_rate_to_euler_pole(wx, wy, wz)

    # Calculate RMS statistics
    re = residuals[::2].flatten()  # residuals east
    rn = residuals[1::2].flatten()  # residuals north

    rms = np.sqrt(np.mean(re**2 + rn**2))

    # Weighted RMS
    try:
        wrms = np.sum((re / sigma_east) ** 2 + (rn / sigma_north) ** 2)
        wrms /= np.sum(1 / sigma_east**2 + 1 / sigma_north**2)
        wrms = np.sqrt(wrms)
    except ZeroDivisionError:
        wrms = np.nan

    # Compile statistics
    statistics = {
        "rms": rms,
        "wrms": wrms,
        "chi_squared": chi2,
        "reduced_chi_squared": reduced_chi2,
        "degrees_of_freedom": dof,
        "parameter_covariance": Q / (1000.0**2),  # (mm/year per km)² to (rad/year)²
        "rotation_rates_rad_per_year": [wx, wy, wz],
    }

    return euler_longitude, euler_latitude, omega, statistics


def get_euler_pole_uncertainty(
    euler_longitude: float,
    euler_latitude: float,
    omega: float,
    parameter_covariance: np.ndarray,
) -> tuple[float, float, float, float]:
    """Calculate Euler pole uncertainty parameters.

    Args:
        euler_longitude: Euler pole longitude in degrees
        euler_latitude: Euler pole latitude in degrees
        omega: Angular velocity in deg/Myr
        parameter_covariance: 3x3 parameter covariance matrix

    Returns:
        Tuple of (max_sigma, min_sigma, azimuth, sigma_omega) representing
        the uncertainty ellipse and angular velocity uncertainty

    """
    # Get rotation matrix for Euler pole location
    rotation_matrix = get_conversion_matrix(euler_longitude, euler_latitude)

    # Convert Euler pole to rotation rates
    wx, wy, wz = euler_pole_to_rotation_rate(euler_longitude, euler_latitude, omega)
    rotation_magnitude = np.sqrt(wx**2 + wy**2 + wz**2)

    # Transform covariance to local ENU frame
    local_covariance = rotation_matrix @ parameter_covariance @ rotation_matrix.T

    # Extract horizontal components
    cov_11 = local_covariance[0, 0]  # East-East
    cov_12 = local_covariance[0, 1]  # East-North
    cov_22 = local_covariance[1, 1]  # North-North

    # Calculate uncertainty ellipse parameters
    discriminant = np.sqrt((cov_11 - cov_22) ** 2 + 4 * cov_12**2)

    max_eigenvalue = 0.5 * (cov_11 + cov_22 + discriminant)
    min_eigenvalue = 0.5 * (cov_11 + cov_22 - discriminant)

    # Convert to angular uncertainties
    max_sigma = np.degrees(np.arctan(np.sqrt(max_eigenvalue) / rotation_magnitude))
    min_sigma = np.degrees(np.arctan(np.sqrt(min_eigenvalue) / rotation_magnitude))

    # Ellipse orientation
    if cov_11 != cov_22:
        azimuth = np.degrees(0.5 * np.arctan(2 * cov_12 / (cov_11 - cov_22)))
    else:
        azimuth = 0.0

    # Angular velocity uncertainty
    sigma_omega = np.degrees(np.sqrt(local_covariance[2, 2])) * MYR_TO_YEAR

    return max_sigma, min_sigma, azimuth, sigma_omega


def model_plate_velocities(
    longitude: float | np.ndarray,
    latitude: float | np.ndarray,
    wx: float,
    wy: float,
    wz: float,
    heights: float | np.ndarray | None = None,
    rotation_covariance: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Model plate motion velocities at given locations using rotation rate components.

    Args:
        longitude: Site longitude(s) in degrees
        latitude: Site latitude(s) in degrees
        wx, wy, wz: Rotation rate components in deg/Myr
        heights: Height(s) above ellipsoid in meters (optional)
        rotation_covariance: 3x3 covariance matrix of rotation parameters in (rad/year)² (optional)

    Returns:
        If rotation_covariance is None:
            Array of modeled velocities [ve, vn] in mm/year, shape (n_sites, 2) or (2,) for single site
        If rotation_covariance is provided:
            Tuple of (modeled_velocities, velocity_uncertainties)
            - modeled_velocities: Array of [ve, vn] in mm/year, shape (n_sites, 2)
            - velocity_uncertainties: Array of [σve, σvn] in mm/year, shape (n_sites, 2)

    """
    # Convert inputs to arrays for consistent handling
    longitude = np.atleast_1d(longitude)
    latitude = np.atleast_1d(latitude)

    if len(longitude) != len(latitude):
        msg = "Longitude and latitude arrays must have the same length"
        raise ValueError(msg)

    n_sites = len(longitude)

    if heights is None:
        heights = np.zeros(n_sites)
    else:
        heights = np.atleast_1d(heights)
        if len(heights) == 1:
            heights = np.full(n_sites, heights[0])
        elif len(heights) != n_sites:
            msg = (
                "Heights array must have same length as coordinates or be a single"
                " value"
            )
            raise ValueError(
                msg
            )

    # Convert rotation rates from deg/Myr to rad/year for internal calculations
    # deg/Myr × (π rad/180°) × (1 Myr/1e6 year) = rad/year
    W = np.array(
        [
            np.radians(wx) / MYR_TO_YEAR,  # rad/year
            np.radians(wy) / MYR_TO_YEAR,  # rad/year
            np.radians(wz) / MYR_TO_YEAR,  # rad/year
        ]
    )

    # Calculate modeled velocities for each site
    modeled_velocities = np.zeros((n_sites, 2))

    # Calculate uncertainties if covariance is provided
    if rotation_covariance is not None:
        uncertainties = np.zeros((n_sites, 2))

    for i, (lon, lat, hgt) in enumerate(zip(longitude, latitude, heights, strict=False)):
        # Get local frame matrix for this site
        local_frame = get_local_frame(lon, lat, hgt)

        # Calculate modeled velocity: v = R × ω
        # local_frame is 3x3, W is 3x1, result is 3x1 [ve, vn, vu]
        velocity_vector = local_frame @ W

        # Extract east and north components (first two elements)
        # Convert from m/year to mm/year
        modeled_velocities[i, 0] = velocity_vector[0] * 1000  # East component (mm/year)
        modeled_velocities[i, 1] = (
            velocity_vector[1] * 1000
        )  # North component (mm/year)

        # Calculate uncertainties if covariance is provided
        if rotation_covariance is not None:
            # Extract East-North components (2x3 matrix)
            J = local_frame[:2, :]  # Jacobian matrix for this site

            # Propagate uncertainty: Σv = J × Σω × J^T
            # where Σω is the rotation parameter covariance matrix
            velocity_covariance = J @ rotation_covariance @ J.T

            # Extract standard deviations and convert from m/year to mm/year
            uncertainties[i, 0] = (
                np.sqrt(velocity_covariance[0, 0]) * 1000
            )  # σve (mm/year)
            uncertainties[i, 1] = (
                np.sqrt(velocity_covariance[1, 1]) * 1000
            )  # σvn (mm/year)

    # Return results based on whether uncertainties were requested
    if rotation_covariance is None:
        # Return single pair for single input, array for multiple inputs
        if n_sites == 1:
            return modeled_velocities[0]
        else:
            return modeled_velocities
    else:
        # Always return arrays when uncertainties are included
        if n_sites == 1:
            return modeled_velocities[0], uncertainties[0]
        else:
            return modeled_velocities, uncertainties


def model_velocities_from_euler_pole(
    longitude: float | np.ndarray,
    latitude: float | np.ndarray,
    euler_longitude: float,
    euler_latitude: float,
    omega: float,
    heights: float | np.ndarray | None = None,
    euler_covariance: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Model plate motion velocities from Euler pole parameters.

    Args:
        longitude: Site longitude(s) in degrees
        latitude: Site latitude(s) in degrees
        euler_longitude: Euler pole longitude in degrees
        euler_latitude: Euler pole latitude in degrees
        omega: Angular velocity in deg/Myr
        heights: Height(s) above ellipsoid in meters (optional)

    Returns:
        Array of modeled velocities [ve, vn] in mm/year
        Shape: (n_sites, 2) or (2,) for single site

    """
    # Convert Euler pole to rotation rate components
    wx, wy, wz = euler_pole_to_rotation_rate(euler_longitude, euler_latitude, omega)

    # Convert from rad/year back to deg/Myr for consistency with predict_plate_motion
    wx_deg_myr = np.degrees(wx) * MYR_TO_YEAR
    wy_deg_myr = np.degrees(wy) * MYR_TO_YEAR
    wz_deg_myr = np.degrees(wz) * MYR_TO_YEAR

    return model_plate_velocities(
        longitude,
        latitude,
        wx_deg_myr,
        wy_deg_myr,
        wz_deg_myr,
        heights,
        euler_covariance,
    )
