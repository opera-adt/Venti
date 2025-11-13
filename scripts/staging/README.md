# OPERA DISP-S1 Data Staging and Correction Tools

Python command-line tools for staging OPERA DISP-S1 displacement products and ancillary data (DEM, LOS geometry, tropospheric corrections, GNSS data).

## Table of Contents

1. [DISP-S1 Products](#disp-s1-products) - Preview and download displacement data
2. [DEM Generation](#dem-generation) - Generate DEM for frame extents
3. [LOS Geometry](#los-geometry) - Generate line-of-sight and incidence angle rasters
4. [Tropospheric Corrections](#tropospheric-corrections) - Download and apply tropo corrections
5. [GNSS Data Processing](#gnss-data-processing) - UNR GNSS data for validation
6. [Installation](#installation)
7. [Typical Workflow](#typical-workflow)

---

## DISP-S1 Products

`disp_cli.py` provides tools for previewing and downloading DISP-S1 displacement products from the ASF DAAC.

### Preview Products

Get summary statistics about available products for a frame:

```bash
python disp_cli.py preview \
    --frame-id 8887 \
    --start 2016-01-01 \
    --end 2017-01-01
```

**Output:**
```
2024-11-06 [INFO] Querying DISP-S1 products for frame 8887
2024-11-06 [INFO] Total files: 150
2024-11-06 [INFO] Reference dates: 45
2024-11-06 [INFO] Secondary dates: 48
2024-11-06 [INFO] Total unique dates: 50
```

#### Preview Options

By default, only summary counts are shown. Use flags to see detailed lists:

| Flag | Description |
|------|-------------|
| `--print-urls` | Print all product URLs |
| `--print-ref-dates` | Print list of reference dates |
| `--print-sec-dates` | Print list of secondary dates |
| `--print-dates` | Print list of all unique dates |

**Example with flags:**
```bash
python disp_cli.py preview \
    --frame-id 8887 \
    --start 2016-01-01 \
    --end 2017-01-01 \
    --print-dates \
    --print-urls
```

#### Shell Redirection

Date lists and URLs are printed to stdout for easy processing:

```bash
# Save dates to file (logging goes to terminal)
python disp_cli.py preview --frame-id 8887 --print-dates > dates.txt

# Get clean list without logging
python disp_cli.py preview --frame-id 8887 --print-urls 2>/dev/null > urls.txt

# Use with wget
python disp_cli.py preview --frame-id 8887 --print-urls 2>/dev/null | wget -i -
```

### Download Products

Download DISP-S1 products for a frame and date range:

```bash
python disp_cli.py download \
    --frame-id 8887 \
    --start 2016-01-01 \
    --end 2017-01-01 \
    --output-dir ./disp_s1 \
    --num-workers 4
```

**Arguments:**
- `--frame-id` (required): OPERA frame identifier
- `--start`: Start date (YYYY-MM-DD or YYYYMMDD format)
- `--end`: End date (YYYY-MM-DD or YYYYMMDD format)
- `--output-dir`: Directory to save products (default: `./disp_s1`)
- `--num-workers`: Number of parallel downloads (default: 2)

**Important Notes:**
- Date queries are based on the **secondary (later) date** of each interferometric pair
- To download a single product, specify the same start and end date
- When start equals end, the date range is automatically expanded by ±1 day
- Only North American frames are currently supported

**Single product download:**
```bash
python disp_cli.py download \
    --frame-id 8887 \
    --start 2016-06-15 \
    --end 2016-06-15
```

---

## DEM Generation

`dem_cli.py` generates DEMs using GLO30 data from `dem_stitcher` for DISP-S1 frame extents.

**Note:** This will be replaced with DISP-STATIC products once available in the ASF DAAC. The DEM generated here may differ from the DEM in DISP-STATIC products, which use GLO30 prepared specifically for NISAR processing.

### Usage

```bash
python dem_cli.py \
    --frame-id 8887 \
    --buffer 10000.0 \
    --output-dir ./dem \
    --use-disp-epsg
```

**Arguments:**
- `--frame-id` (required): OPERA frame identifier
- `--buffer`: Buffer distance in meters around frame extent (default: 10,000)
- `--output-dir`: Output directory (default: `./dem`)
- `--use-disp-epsg`: Also save DEM in frame's native UTM zone (default: False)

**Output:**
- `dem_frame_{frame_id}.tif` - DEM in WGS84 (EPSG:4326)
- `dem_frame_{frame_id}_epsg{code}.tif` - DEM in native UTM (if `--use-disp-epsg` enabled)

**Example:**
```bash
python dem_cli.py \
    --frame-id 8887 \
    --buffer 15000.0 \
    --use-disp-epsg
```

This creates:
- `./dem/dem_frame_8887.tif` (WGS84)
- `./dem/dem_frame_8887_epsg32610.tif` (UTM Zone 10N)

---

## LOS Geometry

`los_cli.py` generates line-of-sight (LOS) unit vectors in East-North-Up (ENU) coordinates and incidence angle rasters from OPERA CSLC-STATIC products.

### Usage

```bash
python los_cli.py \
    --frame-id 8887 \
    --output-dir ./los \
    --inc-out
```

**Arguments:**
- `--frame-id` (required): OPERA frame identifier
- `--output-dir`: Output directory (default: `./los`)
- `--inc-out`: Generate incidence angle raster in addition to LOS ENU (default: True)

**Output:**
- `los_enu_frame_{frame_id}.tif` - 3-band GeoTIFF with LOS components:
  - Band 1: LOS East
  - Band 2: LOS North
  - Band 3: LOS Up
- `los_east.vrt`, `los_north.vrt`, `los_up.vrt` - VRTs for individual components
- `incidence_angle_frame_{frame_id}.tif` - Incidence angle in degrees (if `--inc-out` enabled)

**Process:**
1. Downloads CSLC-STATIC products for the frame's bursts
2. Stitches geometry layers
3. Computes LOS up component as sqrt(1 - east² - north²)
4. Generates 3-band GeoTIFF and component VRTs
5. Optionally computes incidence angle as arccos(los_up)

**Example:**
```bash
python los_cli.py \
    --frame-id 8887 \
    --output-dir ./geometry
```

---

## Tropospheric Corrections

`tropo_cli.py` provides tools to download and apply tropospheric corrections from HRRR weather model data to DISP-S1 displacement products.

### From Single DISP File

Process tropospheric corrections using a single DISP-S1 file:

```bash
python tropo_cli.py file \
    --disp-file ./disp_s1/OPERA_L3_DISP-S1_IW_F08887_VV_20160705T002809Z_20161208T002812Z_v1.0.nc \
    --dem-path ./dem/dem_frame_8887.tif \
    --incidence-angle-path ./los/incidence_angle_frame_8887.tif \
    --output-dir ./tropo \
    --num-workers 4
```

**Arguments:**
- `--disp-file` (required): Path to DISP-S1 NetCDF file
- `--dem-path` (required): Path to DEM file for the frame
- `--incidence-angle-path` (required): Path to incidence angle file
- `--output-dir`: Output directory for tropo products (default: `./tropo`)
- `--num-workers`: Number of parallel workers (default: 2)
- `--dem-hgt-buffer`: Height buffer above DEM max in meters (default: 3000.0)
- `--to-disp-epsg`: Reproject to DISP file's UTM projection (default: True)

### From Stack of DISP Files

Process tropospheric corrections from a directory of DISP-S1 files:

```bash
python tropo_cli.py stack \
    --disp-dir ./disp_s1 \
    --dem-path ./dem/dem_frame_8887.tif \
    --incidence-angle-path ./los/incidence_angle_frame_8887.tif \
    --output-dir ./tropo \
    --frame-id 8887 \
    --num-workers 4
```

**Additional Arguments:**
- `--disp-dir` (required): Path to directory containing DISP-S1 NetCDF files
- `--frame-id`: Specific frame ID to process (required if multiple frames in directory)

### From Database (Frame ID + Date Range)

Process tropospheric corrections using frame ID and date range:

```bash
python tropo_cli.py db \
    --frame-id 8887 \
    --dem-path ./dem/dem_frame_8887.tif \
    --incidence-angle-path ./los/incidence_angle_frame_8887.tif \
    --output-dir ./tropo \
    --start 2016-01-01 \
    --end 2017-01-01 \
    --num-workers 4
```

**Additional Arguments:**
- `--start`: Start date (YYYY-MM-DD or YYYYMMDD format)
- `--end`: End date (YYYY-MM-DD or YYYYMMDD format)

**Output:**
- `tropo_urls.txt` - List of all available tropospheric correction URLs
- `cropped_tropo/` - Cropped tropospheric data for frame extent and times
- `tropo_corrections/` - Applied tropospheric corrections in WGS84
- `tropo_corrections_{epsg}/` - Reprojected corrections in native EPSG (if enabled)

**Process:**
1. Searches ASF for HRRR tropospheric correction products
2. Crops tropospheric data to frame bounds and sensing times
3. Applies corrections using DEM and incidence angle
4. Optionally reprojects to match DISP-S1 UTM grid

---

## GNSS Data Processing

`unr_gnss_processing.py` provides tools to download and process University of Nevada Reno (UNR) gridded GNSS data for validation and correction of DISP-S1 products.

### Download UNR Grid Data

Download UNR gridded GNSS timeseries for a frame:

```bash
python unr_gnss_processing.py download \
    --frame-id 8887 \
    --output-dir ./unr \
    --start 2016-01-01 \
    --end 2024-12-31 \
    --margin-deg 0.5 \
    --plate IGS20 \
    --save-tenv
```

**Arguments:**
- `--frame-id` (required): OPERA frame identifier
- `--output-dir`: Output directory (default: `./unr`)
- `--start`: Start date (YYYY-MM-DD or YYYYMMDD format)
- `--end`: End date (YYYY-MM-DD or YYYYMMDD format)
- `--margin-deg`: Margin in degrees to expand frame bounding box (default: 0.5)
- `--plate`: Reference plate (NA, PA, IGS14, IGS20) (default: IGS20)
- `--version`: UNR grid version (0.1, 0.2) (default: 0.2)
- `--save-tenv`: Save output as .tenv8 files in addition to parquet (default: False)

**Output:**
- `unr_grid_frame{frame_id}.parquet` - Timeseries in parquet format
- `tenv/` - .tenv8 files and lookup table (if `--save-tenv` enabled)

### Compute Velocities

Compute velocities from UNR gridded timeseries using MIDAS:

```bash
# From parquet file
python unr_gnss_processing.py get-velocity \
    --parquet-file ./unr/unr_grid_frame8887.parquet \
    --output-dir ./velocity_output

# From .tenv8 files
python unr_gnss_processing.py get-velocity \
    --tenv-lookup-file ./unr/tenv/grid_latlon_lookup.txt \
    --tenv-dir ./unr/tenv \
    --output-dir ./velocity_output
```

**Arguments:**
- `--parquet-file`: Path to parquet file with timeseries (optional)
- `--tenv-lookup-file`: Path to grid_latlon_lookup.txt (optional)
- `--tenv-dir`: Directory with .tenv8 files (optional)
- `--output-dir`: Output directory (default: `./velocity_output`)

**Note:** Must provide either `--parquet-file` OR both `--tenv-lookup-file` and `--tenv-dir`.

**Output:**
- `velocity_unr_grid.parquet` - Velocity estimates for each station

### Extract Constant Velocity Timeseries

Extract constant velocity timeseries for DISP-S1 products:

#### From Single DISP File

```bash
python unr_gnss_processing.py get-constant-ts-file \
    --disp-file ./disp_s1/OPERA_L3_DISP-S1_IW_F08887_VV_20160705T002809Z_20161208T002812Z_v1.0.nc \
    --velocity-file ./velocity_output/velocity_unr_grid.parquet \
    --output-dir ./constant_ts_output \
    --reference-date 2016-01-01
```

#### From Stack of DISP Files

```bash
python unr_gnss_processing.py get-constant-ts-stack \
    --disp-dir ./disp_s1 \
    --velocity-file ./velocity_output/velocity_unr_grid.parquet \
    --output-dir ./constant_ts_output \
    --frame-id 8887 \
    --reference-date 2016-01-01
```

**Arguments:**
- `--disp-file` / `--disp-dir`: Path to DISP file or directory (required)
- `--velocity-file` (required): Path to velocity parquet file
- `--output-dir`: Output directory (default: `./constant_ts_output`)
- `--reference-date`: Reference date (YYYY-MM-DD or YYYYMMDD)
- `--frame-id`: Specific frame to process (required if multiple frames in stack)

**Output:**
- `constant_ts_{filename}.parquet` - Constant velocity timeseries for single file
- `constant_ts_stack.parquet` - Concatenated timeseries for all files

---

## Installation

All scripts require Python 3.11+ and the following packages:

```bash
pip install numpy tyro rasterio opera-utils asf-search xarray rioxarray pyproj \
    dem-stitcher geopandas pandas pyarrow requests tqdm shapely
```

For GNSS processing, also install:
```bash
pip install geepers
```

Or use the project's existing environment setup.

---

## Typical Workflow

### Basic Workflow (DISP + Ancillary Data)

```bash
# 1. Preview available products
python disp_cli.py preview --frame-id 8887 --start 2016-01-01 --end 2017-01-01

# 2. Download displacement products
python disp_cli.py download --frame-id 8887 --start 2016-01-01 --end 2017-01-01 --output-dir ./data

# 3. Generate DEM for the frame
python dem_cli.py --frame-id 8887 --output-dir ./data --use-disp-epsg

# 4. Generate LOS geometry
python los_cli.py --frame-id 8887 --output-dir ./data
```

### Complete Workflow (With Corrections)

```bash
# 1-4: Same as basic workflow

# 5. Process tropospheric corrections from stack
python tropo_cli.py stack \
    --disp-dir ./data \
    --dem-path ./data/dem_frame_8887.tif \
    --incidence-angle-path ./data/incidence_angle_frame_8887.tif \
    --output-dir ./tropo \
    --frame-id 8887

# 6. Download UNR GNSS data
python unr_cli.py download \
    --frame-id 8887 \
    --output-dir ./unr \
    --start 2016-01-01 \
    --end 2024-12-31

# 7. Compute GNSS velocities
python unr_cli.py get-velocity \
    --parquet-file ./unr/unr_grid_frame8887.parquet \
    --output-dir ./velocity

# 8. Extract constant velocity timeseries
python unr_cli.py get-constant-ts-stack \
    --disp-dir ./data \
    --velocity-file ./velocity/velocity_unr_grid.parquet \
    --output-dir ./constant_ts \
    --frame-id 8887
```

---

## Output Organization

Recommended directory structure:

```
project/
├── disp_s1/                    # Displacement products
│   └── OPERA_L3_DISP-S1_*.nc
├── dem/                        # DEM files
│   ├── dem_frame_8887.tif
│   └── dem_frame_8887_epsg32610.tif
├── los/                        # LOS geometry
│   ├── los_enu_frame_8887.tif
│   ├── incidence_angle_frame_8887.tif
│   └── *.vrt
├── tropo/                      # Tropospheric corrections
│   ├── tropo_urls.txt
│   ├── cropped_tropo/
│   ├── tropo_corrections/
│   └── tropo_corrections_{epsg}/
├── unr/                        # UNR GNSS data
│   ├── unr_grid_frame8887.parquet
│   └── tenv/
├── velocity/                   # GNSS velocities
│   └── velocity_unr_grid.parquet
└── constant_ts/               # Constant velocity timeseries
    └── constant_ts_stack.parquet
```

---

## Date Format Support

All commands support two date formats:
- ISO format: `2016-01-01`
- Compact format: `20160101`

```bash
# These are equivalent
python disp_cli.py preview --frame-id 8887 --start 2016-01-01
python disp_cli.py preview --frame-id 8887 --start 20160101
```

---

## Troubleshooting

### Frame Not Found
- Ensure the frame ID exists in the OPERA database
- Currently only North American frames are supported for DISP-S1

### No Products Found
- Check the date range - products may not exist for all dates
- Date queries are based on secondary dates
- Use `preview` command to see available dates

### Download Errors
- Reduce `--num-workers` if experiencing connection issues
- Ensure sufficient disk space
- Check ASF DAAC status

### Missing LOS Files
- Ensure CSLC-STATIC products exist for all frame bursts
- Check ASF credentials if download fails

### Tropospheric Correction Errors
- Ensure DEM and incidence angle files exist before running tropo corrections
- Check that sensing times extracted from DISP files are within HRRR data availability
- Reduce `--num-workers` if experiencing memory issues

### GNSS Processing Errors
- Check internet connection for UNR grid downloads
- Ensure sufficient disk space for large timeseries
- If processing fails with multiple frames, specify `--frame-id`

---

## Advanced Usage

### Process Multiple Frames

```bash
#!/bin/bash
frames=(8887 8888 8889)
for frame in "${frames[@]}"; do
    python disp_cli.py download --frame-id "$frame" --start 2016-01-01 --end 2017-01-01
    python dem_cli.py --frame-id "$frame" --use-disp-epsg
    python los_cli.py --frame-id "$frame"
    python tropo_cli.py db \
        --frame-id "$frame" \
        --dem-path "./dem/dem_frame_${frame}.tif" \
        --incidence-angle-path "./los/incidence_angle_frame_${frame}.tif" \
        --start 2016-01-01 \
        --end 2017-01-01
done
```

### Extract Specific Dates

```bash
# Get all unique dates
python disp_cli.py preview --frame-id 8887 --print-dates 2>/dev/null > dates.txt

# Process last 10 dates
tail -10 dates.txt | while read date; do
    python disp_cli.py download --frame-id 8887 --start "$date" --end "$date"
done
```

### Automated Pipeline

```bash
# Download products for date range, then stage all ancillary data
python disp_cli.py download --frame-id 8887 --start 2016-01-01 --end 2017-01-01 && \
python dem_cli.py --frame-id 8887 --use-disp-epsg && \
python los_cli.py --frame-id 8887 && \
python tropo_cli.py stack \
    --disp-dir ./disp_s1 \
    --dem-path ./dem/dem_frame_8887.tif \
    --incidence-angle-path ./los/incidence_angle_frame_8887.tif \
    --frame-id 8887
```

---

## Getting Help

Each command provides detailed help:

```bash
python disp_cli.py --help
python disp_cli.py preview --help
python disp_cli.py download --help
python dem_cli.py --help
python los_cli.py --help
python tropo_cli.py --help
python tropo_cli.py file --help
python tropo_cli.py stack --help
python tropo_cli.py db --help
python unr_cli.py --help
python unr_cli.py download --help
python unr_cli.py get-velocity --help
python unr_cli.py get-constant-ts-file --help
python unr_cli.py get-constant-ts-stack --help
```

---

## References

- [OPERA Project](https://www.jpl.nasa.gov/go/opera)
- [ASF DAAC](https://asf.alaska.edu/)
- [opera-utils Documentation](https://github.com/opera-adt/opera-utils)
- [UNR GPS gridded timeseries](https://geodesy.unr.edu/grid_timeseries)
