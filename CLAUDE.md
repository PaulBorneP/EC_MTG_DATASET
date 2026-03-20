# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EarthCARE (EC) ↔ MTG FCI collocation dataset pipeline. Downloads square patches of MTG FCI geostationary satellite imagery centered on EarthCARE polar-orbiting ground tracks. Only the 2–5 FCI chunks intersecting the track are downloaded (~70–175 MB vs ~1.4 GB full disc).

## Commands

```bash
# Run the full pipeline (requires .env with EUMETSAT_KEY and EUMETSAT_SECRET)
python collocation_pipeline.py

# Launch the Dash data viewer (serves on http://localhost:8050)
python viewer.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

**Pipeline modules** (called by `collocation_pipeline.py` in this order):

1. **ec_track.py** — Reads EarthCARE AC_TC_2B HDF5 files, extracts ground tracks (lat/lon/time from `ScienceData/`), filters to MTG disc visibility, generates patch centers spaced along-track, computes bounding boxes, and finds which FCI chunks intersect each patch using WKT polygon footprints.

2. **mtg_download.py** — Authenticates with EUMETSAT Data Store via `eumdac`, searches for FCI products matching the EC overpass time window, and downloads only the specific chunk NetCDF files needed. Credentials loaded from `.env` via `python-dotenv`.

3. **patch_extractor.py** — Handles FCI geostationary projection math (pixel ↔ lat/lon conversion), crops radiance data from chunk files into square patches, builds geolocation arrays, and saves output as NetCDF4. Contains hardcoded FCI grid parameters (IR: 5568px/~2km, VIS: 11136px/~1km).

**Viewer** (`viewer.py`): Dash web app showing world map (via `earthcarekit`), interactive context map (Plotly/Scattermap), MTG radiance patch (matplotlib), and EC curtain cross-section side by side. Reads output patches from `DATA/patches/` and EC files from `DATA/EC/`.

## Data Layout

- `DATA/EC/` — Input EarthCARE HDF5 files (AC_TC_2B product)
- `DATA/MTG/` — Downloaded FCI chunk NetCDF files (cache, reused across runs)
- `DATA/patches/` — Output: per-patch NetCDF files + `index.csv`
- `FCI_chunks.wkt` — Precomputed footprint polygons for 40 FCI chunks (CSV: chunk_id,WKT)
- `config.yaml` — Pipeline parameters (patch size, channels, time thresholds, paths)

## Key Technical Details

- FCI chunk filenames encode scan times as `OPE_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS` — parsed by regex in `mtg_download.py`
- Chunk 0041 is always downloaded (trailer chunk) but skipped during extraction
- EarthCARE time is seconds since 2000-01-01 epoch
- Patch pixel ranges are 1-based in FCI convention, converted to 0-based for array indexing in `patch_extractor.py`
- The viewer requires `earthcarekit` (not in requirements.txt) for curtain/map figures

always use earthcare conda env