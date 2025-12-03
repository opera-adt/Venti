"""UNR GNSS data processing for OPERA DISP-S1 corrections.

Extract GNSS displacement data from University of Nevada Reno (UNR) for use
with OPERA DISP-S1 interferograms. Supports downloading gridded data, computing
velocities, and extracting constant velocity timeseries.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from geepers.gps_sources import UnrGridSource
from geepers.midas import midas
from geepers.utils import decimal_year_to_datetime
from opera_utils import get_frame_geojson
from tqdm import tqdm
from utils import (
    datetime_to_decimal_year,
    extract_frame_id_from_filename,
    extract_sensing_times_from_file,
    parse_date,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Download functionality
# ==============================================================================


def download_unr_grid(
    frame_id: int,
    output_dir: Path,
    start: datetime | None = None,
    end: datetime | None = None,
    margin_deg: float = 0.5,
    plate: Literal["NA", "PA", "IGS14", "IGS20"] = "IGS20",
    version: Literal["0.1", "0.2"] = "0.2",
    save_tenv: bool = False,
) -> None:
    """Download UNR gridded GNSS timeseries for a given frame.

    Parameters
    ----------
    frame_id : int
        OPERA frame identifier.
    output_dir : Path
        Output directory for downloaded data.
    start : datetime or None, optional
        Start date for timeseries. If None, downloads from beginning.
    end : datetime or None, optional
        End date for timeseries. If None, downloads until present.
    margin_deg : float, default=0.5
        Margin in degrees to expand frame bounding box.
    plate : {"NA", "PA", "IGS14", "IGS20"}, default="IGS20"
        Reference plate for velocity computation.
    version : {"0.1", "0.2"}, default="0.2"
        UNR grid version to download.
    save_tenv : bool, default=False
        Save output as .tenv8 files in addition to parquet.

    """
    logger.info(
        f"Downloading UNR gridded GNSS timeseries for frame {frame_id} "
        f"from {start or 'beginning'} to {end or 'present'}"
    )

    selected_frame = get_frame_geojson([frame_id], as_geodataframe=True)
    frame_bounds = tuple(selected_frame.bounds.values[0])

    extended_bbox = (
        frame_bounds[0] - margin_deg,
        frame_bounds[1] - margin_deg,
        frame_bounds[2] + margin_deg,
        frame_bounds[3] + margin_deg,
    )

    grid = UnrGridSource(version=version)
    ts_grid_df = grid.timeseries_many(bbox=extended_bbox)
    ts_grid_df["date"] = pd.to_datetime(ts_grid_df["date"])

    output_dir.mkdir(parents=True, exist_ok=True)

    _save_parquet(ts_grid_df, frame_id, plate, output_dir)

    if save_tenv:
        _save_as_tenv_files(ts_grid_df, output_dir / "tenv", plate, version)


def _save_parquet(
    df: pd.DataFrame,
    frame_id: int,
    plate: str,
    output_dir: Path,
) -> None:
    """Save timeseries DataFrame to parquet with metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries data to save.
    frame_id : int
        Frame identifier for metadata.
    plate : str
        Reference plate for metadata.
    output_dir : Path
        Output directory.

    """
    df_no_geom = df.drop(columns=["geometry"], errors="ignore")
    table = pa.Table.from_pandas(df_no_geom)

    metadata = {
        "frame_id": str(frame_id),
        "source": "UNR grid",
        "description": "Time series of east/north/up displacements and uncertainties",
        "reference_frame": plate,
    }

    existing_metadata = table.schema.metadata or {}
    encoded_metadata = {k: v.encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata({**existing_metadata, **encoded_metadata})

    output_file = output_dir / f"unr_grid_frame{frame_id}.parquet"
    pq.write_table(table, output_file)
    logger.info(f"Saved parquet to {output_file}")


def _save_as_tenv_files(
    df: pd.DataFrame,
    output_dir: Path,
    plate: str,
    version: str,
) -> None:
    """Save timeseries as .tenv8 files.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries data to save.
    output_dir : Path
        Output directory for .tenv8 files.
    plate : str
        Reference plate name.
    version : str
        UNR grid version.

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    latlon_df = df.groupby("id")[["lon", "lat"]].first().reset_index()

    if version == "0.2":
        latlon_df["lon"] = (latlon_df["lon"] + 360) % 360

    latlon_df.to_csv(
        output_dir / "grid_latlon_lookup.txt",
        sep=" ",
        index=False,
        header=False,
        float_format="%.6f",
    )

    for station_id in tqdm(df["id"].unique(), desc="Saving .tenv8 files"):
        station_df = df[df["id"] == station_id].copy()
        _save_single_tenv8(station_df, output_dir, plate)


def _save_single_tenv8(
    df: pd.DataFrame,
    output_dir: Path,
    plate: str,
) -> None:
    """Save single station timeseries as .tenv8 file.

    Parameters
    ----------
    df : pd.DataFrame
        Single station timeseries data.
    output_dir : Path
        Output directory.
    plate : str
        Reference plate name.

    """
    station_id = df["id"].values[0]

    df = df.copy()
    df.loc[:, ["east", "north", "up"]] *= 1000
    df["zero_column"] = 0
    df["date"] = df["date"].apply(datetime_to_decimal_year)

    cols_to_drop = [
        "id",
        "corr_en",
        "corr_eu",
        "corr_nu",
        "lon",
        "lat",
        "alt",
        "geometry",
    ]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)

    output_file = output_dir / f"{int(station_id):06d}_{plate}.tenv8"
    df.to_csv(output_file, sep=" ", index=False, header=False, float_format="%.4f")


# ==============================================================================
# Velocity computation
# ==============================================================================


def compute_velocities(
    parquet_file: Path | None = None,
    tenv_lookup_file: Path | None = None,
    tenv_dir: Path | None = None,
    output_dir: Path = Path("./velocity_output"),
) -> pd.DataFrame:
    """Compute velocities from UNR gridded GNSS timeseries.

    Parameters
    ----------
    parquet_file : Path or None, optional
        Path to parquet file with timeseries. If None, must provide tenv files.
    tenv_lookup_file : Path or None, optional
        Path to grid_latlon_lookup.txt file. Required if using tenv files.
    tenv_dir : Path or None, optional
        Directory containing .tenv8 files. Required if using tenv files.
    output_dir : Path, default=Path("./velocity_output")
        Output directory for velocity results.

    Returns
    -------
    pd.DataFrame
        DataFrame with velocity estimates for each station.

    Raises
    ------
    ValueError
        If neither parquet_file nor tenv files are provided.

    """
    if parquet_file is not None:
        logger.info(f"Loading timeseries from {parquet_file}")
        df, _ = _load_parquet(parquet_file)
    elif tenv_lookup_file is not None and tenv_dir is not None:
        logger.info(f"Loading timeseries from .tenv8 files in {tenv_dir}")
        df = _load_tenv_files(tenv_lookup_file, tenv_dir)
    else:
        msg = "Must provide either parquet_file or both tenv_lookup_file and tenv_dir"
        raise ValueError(msg)

    logger.info("Computing velocities using MIDAS")
    rates_dfs = [
        _compute_station_velocity(df[df.id == station_id])
        for station_id in tqdm(df.id.unique(), desc="Computing velocities")
    ]
    rates_df = pd.concat(rates_dfs, ignore_index=True)

    for station_id in rates_df.id.unique():
        row = df[df["id"] == station_id].iloc[0]
        for col in ("lon", "lat", "alt"):
            rates_df.loc[rates_df["id"] == station_id, col] = row[col]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "velocity_unr_grid.parquet"
    rates_df.to_parquet(output_file)
    logger.info(f"Saved velocities to {output_file}")

    return rates_df


def _compute_station_velocity(station_df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity for single GNSS station using MIDAS.

    Parameters
    ----------
    station_df : pd.DataFrame
        Timeseries data for single station with 'date', 'east', 'north', 'up'.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with velocity and uncertainty estimates.

    """
    years = (station_df["date"] - station_df["date"].min()).dt.total_seconds() / (
        365.25 * 24 * 3600
    )

    components = ["east", "north", "up"]
    velocities = {}
    residuals = {}

    for comp in components:
        values = station_df[comp].to_numpy()
        midas_res = midas(times=years.to_numpy(), values=values)
        velocities[f"velocity_{comp}"] = midas_res.velocity
        velocities[f"velocity_uncertainty_{comp}"] = midas_res.velocity_uncertainty
        residuals[f"residuals_{comp}"] = [midas_res.residuals]

    data = {
        "id": station_df["id"].values[0],
        "dates": [station_df["date"].values],
        **velocities,
        **residuals,
        "duration": years.max(),
    }

    return pd.DataFrame(data)


def _load_parquet(file_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load parquet file and metadata.

    Parameters
    ----------
    file_path : Path
        Path to parquet file.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        Tuple of (DataFrame with geometry, metadata dictionary).

    """
    df = pd.read_parquet(file_path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.lon, y=df.lat))

    meta = pq.read_metadata(file_path).metadata
    metadata_dict = {k.decode(): v.decode() for k, v in meta.items()}

    return gdf, metadata_dict


def _load_tenv_files(
    latlon_lookup: Path,
    tenv_dir: Path,
    version: str = "0.2",
) -> pd.DataFrame:
    """Load timeseries from .tenv8 files.

    Parameters
    ----------
    latlon_lookup : Path
        Path to grid_latlon_lookup.txt file.
    tenv_dir : Path
        Directory containing .tenv8 files.
    version : str, default="0.2"
        UNR grid version.

    Returns
    -------
    pd.DataFrame
        Concatenated timeseries for all stations.

    """
    unr = UnrGridSource()

    df = pd.read_csv(latlon_lookup, sep=r"\s+", names=["id", "lon", "lat"])

    if version == "0.2":
        df["lon"] = ((df["lon"] + 180) % 360) - 180

    df["alt"] = 0.0

    gdf_stations = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326",
    )

    def _load_single_tenv(path: Path) -> pd.DataFrame:
        filename = path.stem
        grid_id = filename.split("_")[0]
        row = gdf_stations[gdf_stations["id"] == int(grid_id)]

        ts_df = unr._read_data_file(path)
        ts_df["date"] = ts_df["decimal_year"].apply(decimal_year_to_datetime)
        ts_df.loc[:, ["east", "north", "up"]] /= 1000

        ts_df["corr_en"] = 0.0
        ts_df["corr_eu"] = 0.0
        ts_df["corr_nu"] = 0.0

        for col in ("id", "lon", "lat", "alt", "geometry"):
            ts_df[col] = row.iloc[0][col]

        return ts_df.loc[
            :,
            [
                "id",
                "date",
                "east",
                "north",
                "up",
                "sigma_east",
                "sigma_north",
                "sigma_up",
                "corr_en",
                "corr_eu",
                "corr_nu",
                "lon",
                "lat",
                "alt",
                "geometry",
            ],
        ]

    dfs = [
        _load_single_tenv(path)
        for path in tqdm(
            Path(tenv_dir).rglob("*.tenv8"),
            desc="Loading .tenv8 files",
        )
    ]

    return pd.concat(dfs, ignore_index=True)


# ==============================================================================
# Constant velocity timeseries extraction
# ==============================================================================


def compute_displacement_from_velocity(
    velocity_df: pd.DataFrame,
    station_id: int,
    ref_date: pd.Timestamp,
    sec_date: pd.Timestamp,
    scale: float = 1000.0,
) -> dict[str, float | int]:
    """Compute displacement from velocity between two dates.

    Parameters
    ----------
    velocity_df : pd.DataFrame
        DataFrame containing velocity columns.
    station_id : int
        Station identifier.
    ref_date : pd.Timestamp
        Reference date.
    sec_date : pd.Timestamp
        Secondary date.
    scale : float, default=1000.0
        Scale factor for displacements (e.g., 1000 for mm to m).

    Returns
    -------
    dict[str, float | int]
        Dictionary with station_id, dates, displacements, and uncertainties.

    Raises
    ------
    ValueError
        If station not found in velocity DataFrame.

    """
    station_df = velocity_df[velocity_df["id"] == station_id]

    if len(station_df) == 0:
        msg = f"Station {station_id} not found in velocity data"
        raise ValueError(msg)

    dt = sec_date - ref_date
    dt_sec = dt / np.timedelta64(1, "s")
    dt_years = dt_sec / (365.25 * 24 * 3600)

    result = {
        "id": station_id,
        "ref_date": datetime_to_decimal_year(ref_date),
        "sec_date": datetime_to_decimal_year(sec_date),
        "east": station_df["velocity_east"].values[0] * scale * dt_years,
        "north": station_df["velocity_north"].values[0] * scale * dt_years,
        "up": station_df["velocity_up"].values[0] * scale * dt_years,
        "sigma_east": (
            np.abs(dt_years) * station_df["velocity_uncertainty_east"].values[0] * scale
        ),
        "sigma_north": (
            np.abs(dt_years)
            * station_df["velocity_uncertainty_north"].values[0]
            * scale
        ),
        "sigma_up": (
            np.abs(dt_years) * station_df["velocity_uncertainty_up"].values[0] * scale
        ),
    }

    return result


def extract_constant_ts_from_file(
    disp_file: Path,
    velocity_file: Path,
    output_dir: Path,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Extract constant velocity timeseries for single DISP-S1 file.

    Parameters
    ----------
    disp_file : Path
        Path to DISP-S1 NetCDF file.
    velocity_file : Path
        Path to velocity parquet file.
    output_dir : Path
        Output directory for results.
    reference_date : datetime or None, optional
        Reference date. If None, uses first date from file.

    Returns
    -------
    pd.DataFrame
        DataFrame with displacements for all stations at file dates.

    """
    logger.info(f"Extracting constant velocity timeseries from {disp_file}")

    vel_df = pd.read_parquet(velocity_file)
    sensing_times = extract_sensing_times_from_file(disp_file)

    ref_date = reference_date if reference_date is not None else sensing_times[0]
    sec_date = sensing_times[1]

    results = []
    for station_id in vel_df.id.unique():
        ref_disp = compute_displacement_from_velocity(
            vel_df, station_id, ref_date=ref_date, sec_date=ref_date
        )
        sec_disp = compute_displacement_from_velocity(
            vel_df, station_id, ref_date=ref_date, sec_date=sec_date
        )

        df_station = pd.concat(
            [pd.DataFrame([ref_disp]), pd.DataFrame([sec_disp])],
            ignore_index=True,
        )

        meta = vel_df.loc[vel_df.id == station_id, ["lon", "lat", "alt"]].iloc[0]
        df_station["lon"] = meta["lon"]
        df_station["lat"] = meta["lat"]
        df_station["alt"] = meta["alt"]

        results.append(df_station)

    result_df = pd.concat(results, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"constant_ts_{disp_file.stem}.parquet"
    result_df.to_parquet(output_file)
    logger.info(f"Saved constant velocity timeseries to {output_file}")

    return result_df


def extract_constant_ts_from_stack(
    disp_dir: Path,
    velocity_file: Path,
    output_dir: Path,
    reference_date: datetime | None = None,
    frame_id: int | None = None,
) -> pd.DataFrame:
    """Extract constant velocity timeseries for stack of DISP-S1 files.

    Parameters
    ----------
    disp_dir : Path
        Directory containing DISP-S1 NetCDF files.
    velocity_file : Path
        Path to velocity parquet file.
    output_dir : Path
        Output directory for results.
    reference_date : datetime or None, optional
        Reference date. If None, uses first date from each file.
    frame_id : int or None, optional
        Process only files for this frame. If None and multiple frames exist,
        raises error.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with displacements for all files.

    Raises
    ------
    ValueError
        If multiple frames found and frame_id not specified.

    """
    logger.info(f"Extracting constant velocity timeseries from directory {disp_dir}")

    disp_list = list(Path(disp_dir).glob("OPERA_L3_DISP-S1*.nc"))
    logger.info(f"Found {len(disp_list)} DISP-S1 files")

    frame_id_list = [extract_frame_id_from_filename(f) for f in disp_list]
    unique_frame_ids = np.unique(frame_id_list)

    if frame_id is not None:
        disp_list = [
            f
            for f, fid in zip(disp_list, frame_id_list, strict=True)
            if fid == frame_id
        ]
        logger.info(f"Filtered to {len(disp_list)} files for frame {frame_id}")
    elif len(unique_frame_ids) > 1:
        msg = (
            f"Multiple frame IDs detected: {unique_frame_ids}. "
            "Specify frame_id parameter."
        )
        raise ValueError(msg)

    results = []
    for file in tqdm(disp_list, desc="Processing files"):
        df = extract_constant_ts_from_file(
            file, velocity_file, output_dir, reference_date
        )
        results.append(df)

    result_df = pd.concat(results, ignore_index=True)

    output_file = output_dir / "constant_ts_stack.parquet"
    result_df.to_parquet(output_file)
    logger.info(f"Saved concatenated timeseries to {output_file}")

    return result_df


# ==============================================================================
# CLI dataclasses
# ==============================================================================


@dataclass
class Download:
    """Download UNR gridded GNSS timeseries for a frame.

    Attributes
    ----------
    frame_id : int
        OPERA frame identifier.
    output_dir : Path, default=Path("./unr")
        Output directory for downloaded data.
    start : str or None, optional
        Start date (YYYY-MM-DD or YYYYMMDD). If None, downloads from beginning.
    end : str or None, optional
        End date (YYYY-MM-DD or YYYYMMDD). If None, downloads until present.
    margin_deg : float, default=0.5
        Margin in degrees to expand frame bounding box.
    plate : {"NA", "PA", "IGS14", "IGS20"}, default="IGS20"
        Reference plate for velocity computation.
    version : {"0.1", "0.2"}, default="0.2"
        UNR grid version to download.
    save_tenv : bool, default=False
        Save output as .tenv8 files in addition to parquet.

    Examples
    --------
    Download data for frame 1234:

        $ python unr_cli.py download --frame-id 1234

    """

    frame_id: int
    output_dir: Path = Path("./unr")
    start: str | None = None
    end: str | None = None
    margin_deg: float = 0.5
    plate: Literal["NA", "PA", "IGS14", "IGS20"] = "IGS20"
    version: Literal["0.1", "0.2"] = "0.2"
    save_tenv: bool = False

    def __call__(self) -> None:
        """Execute download command."""
        start_dt = parse_date(self.start) if self.start else None
        end_dt = parse_date(self.end) if self.end else None

        download_unr_grid(
            frame_id=self.frame_id,
            output_dir=self.output_dir,
            start=start_dt,
            end=end_dt,
            margin_deg=self.margin_deg,
            plate=self.plate,
            version=self.version,
            save_tenv=self.save_tenv,
        )


@dataclass
class GetVelocity:
    r"""Compute velocities from UNR gridded GNSS timeseries.

    Attributes
    ----------
    parquet_file : Path or None, optional
        Path to parquet file with timeseries. Required if not using tenv files.
    tenv_lookup_file : Path or None, optional
        Path to grid_latlon_lookup.txt. Required if using tenv files.
    tenv_dir : Path or None, optional
        Directory with .tenv8 files. Required if using tenv files.
    output_dir : Path, default=Path("./velocity_output")
        Output directory for velocity results.

    Examples
    --------
    Compute velocities from parquet:

        $ python unr_cli.py get-velocity --parquet-file ./unr/data.parquet

    Compute velocities from tenv files:

        $ python unr_cli.py get-velocity \\
            --tenv-lookup-file ./unr/tenv/grid_latlon_lookup.txt \\
            --tenv-dir ./unr/tenv

    """

    parquet_file: Path | None = None
    tenv_lookup_file: Path | None = None
    tenv_dir: Path | None = None
    output_dir: Path = Path("./velocity_output")

    def __call__(self) -> None:
        """Execute velocity computation."""
        compute_velocities(
            parquet_file=self.parquet_file,
            tenv_lookup_file=self.tenv_lookup_file,
            tenv_dir=self.tenv_dir,
            output_dir=self.output_dir,
        )


@dataclass
class GetConstantTSFile:
    r"""Extract constant velocity timeseries from single DISP-S1 file.

    Attributes
    ----------
    disp_file : Path
        Path to DISP-S1 NetCDF file.
    velocity_file : Path
        Path to velocity parquet file.
    output_dir : Path, default=Path("./constant_ts_output")
        Output directory for results.
    reference_date : str or None, optional
        Reference date (YYYY-MM-DD or YYYYMMDD). If None, uses first date from file.

    Examples
    --------
    Extract timeseries from file:

        $ python unr_cli.py get-constant-ts file \\
            --disp-file ./data/OPERA_L3_DISP-S1_20230101.nc \\
            --velocity-file ./velocity_output/velocity_unr_grid.parquet

    """

    disp_file: Path
    velocity_file: Path
    output_dir: Path = Path("./constant_ts_output")
    reference_date: str | None = None

    def __call__(self) -> None:
        """Execute constant velocity extraction from file."""
        ref_dt = parse_date(self.reference_date) if self.reference_date else None

        extract_constant_ts_from_file(
            disp_file=self.disp_file,
            velocity_file=self.velocity_file,
            output_dir=self.output_dir,
            reference_date=ref_dt,
        )


@dataclass
class GetConstantTSStack:
    r"""Extract constant velocity timeseries from stack of DISP-S1 files.

    Attributes
    ----------
    disp_dir : Path
        Directory containing DISP-S1 NetCDF files.
    velocity_file : Path
        Path to velocity parquet file.
    output_dir : Path, default=Path("./constant_ts_output")
        Output directory for results.
    reference_date : str or None, optional
        Reference date (YYYY-MM-DD or YYYYMMDD). If None,
        uses first date from each file.
    frame_id : int or None, optional
        Process only files for this frame. If None and multiple frames exist,
        raises error.

    Examples
    --------
    Extract timeseries from stack:

        $ python unr_cli.py get-constant-ts stack \\
            --disp-dir ./data \\
            --velocity-file ./velocity_output/velocity_unr_grid.parquet \\
            --frame-id 1234

    """

    disp_dir: Path
    velocity_file: Path
    output_dir: Path = Path("./constant_ts_output")
    reference_date: str | None = None
    frame_id: int | None = None

    def __call__(self) -> None:
        """Execute constant velocity extraction from stack."""
        ref_dt = parse_date(self.reference_date) if self.reference_date else None

        extract_constant_ts_from_stack(
            disp_dir=self.disp_dir,
            velocity_file=self.velocity_file,
            output_dir=self.output_dir,
            reference_date=ref_dt,
            frame_id=self.frame_id,
        )


# ==============================================================================
# CLI entry point
# ==============================================================================


def main() -> None:
    """Run CLI with download, get_velocity, and get_constant_ts commands."""
    import tyro

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    tyro.extras.subcommand_cli_from_dict(
        {
            "download": Download,
            "get-velocity": GetVelocity,
            "get-constant-ts-file": GetConstantTSFile,
            "get-constant-ts-stack": GetConstantTSStack,
        }
    )()


if __name__ == "__main__":
    main()
