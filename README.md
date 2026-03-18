# EarthCARE ↔ MTG FCI Collocation Dataset

Pipeline to extract square patches of MTG FCI geostationary imagery centered on EarthCARE polar-orbiting ground tracks. Only the 2–5 FCI chunks intersecting the track are downloaded instead of the full disc (~70–175 MB vs ~1.4 GB).

## Quick Start

```bash
# 1. Set up credentials in .env
EUMETSAT_KEY="your_key"
EUMETSAT_SECRET="your_secret"

# 2. Place EarthCARE AC_TC_2B HDF5 files in DATA/EC/

# 3. Run the pipeline
python collocation_pipeline.py
```

## Configuration

Edit `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size_km` | 100 | Side length of square patches in km |
| `fci_channels` | `[ir_105, vis_06]` | FCI channels to extract |
| `fci_collection` | `EO:EUM:DAT:0662` | EUMETSAT collection ID (FDHSI) |
| `max_time_diff_minutes` | 5 | Max allowed EC↔MTG time difference |
| `ec_data_dir` | `DATA/EC` | Input EarthCARE files |
| `mtg_data_dir` | `DATA/MTG` | MTG chunk download cache |
| `patch_output_dir` | `DATA/patches` | Output patches |
| `chunk_wkt_file` | `FCI_chunks.wkt` | FCI chunk footprint polygons |

## Architecture

```
ec_track.py              # EarthCARE track extraction & patch geometry
mtg_download.py          # MTG chunk selection & EUMETSAT download
patch_extractor.py       # Local cropping from chunks → square patches
collocation_pipeline.py  # Orchestration script
config.yaml              # Pipeline configuration
FCI_chunks.wkt           # 40 chunk footprint polygons (precomputed)
```

## Pipeline Flow

1. Authenticate with EUMETSAT Data Store (`eumdac`)
2. For each EarthCARE HDF5 file:
   - Extract ground track (lat, lon, time) from `ScienceData/`
   - Filter to MTG disc (angular distance ≤ 75° from subsatellite point)
   - Generate patch centers spaced by `patch_size_km` along the track
   - For each center, compute bounding box and find intersecting FCI chunks
   - Download only the needed chunks from the closest-in-time MTG product
   - Crop patches from chunks and save as NetCDF with geolocation
3. Write `DATA/patches/index.csv` summary

## Output

```
DATA/patches/
  index.csv
  EC_MTG_patch_07470A_000_20250921T003743.nc
  EC_MTG_patch_07470A_001_20250921T003855.nc
  ...
```

Each patch NetCDF contains:
- `effective_radiance_{channel}(y, x)` — radiance data per channel
- `latitude(y, x)`, `longitude(y, x)` — coordinate arrays
- Global attributes: EC source file, MTG product ID, time difference, patch center/size

The `index.csv` columns: `ec_file, patch_idx, center_lat, center_lon, ec_time, mtg_time, time_diff_s, mtg_chunks, patch_file`

## Dependencies

`h5py`, `netCDF4`, `numpy`, `xarray`, `shapely`, `pyproj`, `eumdac`, `pyyaml`, `tqdm`

## Data Sources

- **EarthCARE**: AC_TC_2B target classification product (HDF5)
- **MTG FCI**: Level 1c Normal Resolution (FDHSI) from [EUMETSAT Data Store](https://data.eumetsat.int/product/EO:EUM:DAT:0662)
- **Chunk footprints**: Precomputed WKT file mapping each of the 40 FCI chunks to geographic polygons
