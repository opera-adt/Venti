# OPERA DISP-S1 Data Staging Tools

Python command-line tools for staging OPERA DISP-S1 displacement products and ancillary data (DEM, LOS geometry).

## Table of Contents

1. [DISP-S1 Products](#disp-s1-products) - Preview and download displacement data
2. [DEM Generation](#dem-generation) - Generate DEM for frame extents
3. [LOS Geometry](#los-geometry) - Generate line-of-sight and incidence angle rasters

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

## Installation

All scripts require Python 3.11+ and the following packages:

```bash
pip install numpy tyro rasterio opera-utils asf-search xarray rioxarray pyproj dem-stitcher
```

Or use the project's existing environment setup.

---

## Standalone Run with `uv`

You can run the scripts without a virtual environment using `uv`:

### DISP Preview/Download

```bash
uv run \
    --python=3.12 \
    --with "numpy" \
    --with tyro \
    --with opera-utils \
    python disp_cli.py preview \
    --frame-id 8887 \
    --start 2016-01-01 \
    --end 2017-01-01
```

### DEM Generation

```bash
uv run \
    --python=3.12 \
    --with "numpy" \
    --with tyro \
    --with xarray \
    --with rioxarray \
    --with pyproj \
    --with opera-utils \
    --with dem-stitcher \
    python dem_cli.py \
    --frame-id 8887 \
    --output-dir ./dem \
    --use-disp-epsg
```

### LOS Geometry

```bash
uv run \
    --python=3.12 \
    --with "numpy" \
    --with tyro \
    --with rasterio \
    --with opera-utils \
    --with asf-search \
    python los_cli.py \
    --frame-id 8887 \
    --output-dir ./los
```

**Note:** `uv` may have issues with some dependencies like `pyproj`. If you encounter problems, use a traditional virtual environment instead.

---

## Date Format Support

Both `preview` and `download` commands support two date formats:
- ISO format: `2016-01-01`
- Compact format: `20160101`

```bash
# These are equivalent
python disp_cli.py preview --frame-id 8887 --start 2016-01-01
python disp_cli.py preview --frame-id 8887 --start 20160101
```

---

## Typical Workflow

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

---

## Output Organization

Recommended directory structure:

```
project/
├── disp_s1/           # Displacement products
│   └── frame_8887/
├── dem/               # DEM files
│   ├── dem_frame_8887.tif
│   └── dem_frame_8887_epsg32610.tif
└── los/               # LOS geometry
    ├── los_enu_frame_8887.tif
    ├── incidence_angle_frame_8887.tif
    ├── los_east.vrt
    ├── los_north.vrt
    └── los_up.vrt
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

---

## Advanced Usage

### Process Multiple Frames

```bash
#!/bin/bash
frames=(8887 8888 8889)
for frame in "${frames[@]}"; do
    python disp_cli.py download --frame-id "$frame" --start 2016-01-01 --end 2017-01-01
    python dem_cli.py --frame-id "$frame"
    python los_cli.py --frame-id "$frame"
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
# Download products for date range, then stage ancillary data
python disp_cli.py download --frame-id 8887 --start 2016-01-01 --end 2017-01-01 && \
python dem_cli.py --frame-id 8887 --use-disp-epsg && \
python los_cli.py --frame-id 8887
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
```

---

## References

- [OPERA Project](https://www.jpl.nasa.gov/go/opera)
- [ASF DAAC](https://asf.alaska.edu/)
- [opera-utils Documentation](https://github.com/opera-adt/opera-utils)
