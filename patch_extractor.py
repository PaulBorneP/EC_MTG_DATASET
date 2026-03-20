"""Local cropping of FCI chunks into square patches using satpy."""

import os
import gc
import logging
import numpy as np
import netCDF4 as nc
import dask
from satpy import Scene

from ec_track import compute_patch_bbox

logger = logging.getLogger(__name__)


def extract_patch(chunk_files, center_lat, center_lon, size_km, channels):
    """Extract a square patch from downloaded FCI chunks.

    Uses satpy's fci_l1c_nc reader for projection-correct geolocation,
    reading scale/offset parameters directly from each chunk file.

    Parameters
    ----------
    chunk_files : dict {chunk_id: filepath}
    center_lat, center_lon : float
    size_km : float
    channels : list of str  (e.g. ["ir_105", "vis_06"])

    Returns
    -------
    dict with keys:
        "data" : {channel: 2D numpy array}
        "lat"  : 2D array of latitudes
        "lon"  : 2D array of longitudes
    """
    # Geographic bounding box for the patch
    bbox = compute_patch_bbox(center_lat, center_lon, size_km)
    # bbox = (min_lon, min_lat, max_lon, max_lat) — matches satpy ll_bbox

    # Collect BODY chunk file paths (exclude TRAIL/0041) # why ??
    filepaths = [fp for cid, fp in chunk_files.items()
                 if cid != "0041" and os.path.isfile(fp)]

    if not filepaths:
        raise ValueError("No valid chunk files provided")

    # Verify each chunk file can actually be opened — catches truncated downloads
    valid_filepaths = []
    for fp in filepaths:
        try:
            with nc.Dataset(fp, "r"):
                pass
            valid_filepaths.append(fp)
        except Exception:
            logger.warning("Corrupt/truncated chunk, deleting for re-download: %s", fp)
            try:
                os.remove(fp)
            except OSError:
                pass
    if not valid_filepaths:
        raise ValueError("All chunk files are corrupt or missing")
    filepaths = valid_filepaths

    # netCDF4/HDF5 are not thread-safe; force synchronous execution throughout
    with dask.config.set(scheduler='synchronous'):
        scn = Scene(filenames=filepaths, reader='fci_l1c_nc')

        # Load available channels with radiance calibration
        available = set(scn.available_dataset_names())
        to_load = [ch for ch in channels if ch in available]
        if not to_load:
            raise ValueError(f"None of {channels} found in chunk files")

        scn.load(to_load, calibration='radiance')

        # Crop to patch geographic extent
        cropped = scn.crop(ll_bbox=bbox)

        result = {"data": {}}
        ref_channel = to_load[0]

        for channel in to_load:
            ds = cropped[channel]
            data = np.array(ds.values, dtype=np.float32)
            result["data"][channel] = data

        # Geolocation from reference channel area definition
        ref_area = cropped[ref_channel].attrs['area']
        lons, lats = ref_area.get_lonlats()
        result["lat"] = np.where(np.isfinite(lats), lats, np.nan).astype(np.float32)
        result["lon"] = np.where(np.isfinite(lons), lons, np.nan).astype(np.float32)

        # Explicitly release Scene (closes HDF5 file handles) before returning
        del scn, cropped

    # Force GC to flush HDF5 file handles before the next patch is processed
    gc.collect()

    return result


def save_patch(patch_data, metadata, output_path):
    """Save extracted patch as NetCDF4.

    Parameters
    ----------
    patch_data : dict from extract_patch()
    metadata : dict with keys like ec_file, mtg_product, time_diff_s,
               center_lat, center_lon, patch_size_km
    output_path : str
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ds = nc.Dataset(output_path, "w", format="NETCDF4")

    # Dimensions from the lat/lon arrays
    ny, nx = patch_data["lat"].shape
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)

    # Coordinate variables
    lat_var = ds.createVariable("latitude", "f4", ("y", "x"), zlib=True)
    lat_var[:] = patch_data["lat"]
    lat_var.units = "degrees_north"

    lon_var = ds.createVariable("longitude", "f4", ("y", "x"), zlib=True)
    lon_var[:] = patch_data["lon"]
    lon_var.units = "degrees_east"

    # Data variables
    for channel, data in patch_data["data"].items():
        # Channels may have different grid sizes; store each at native res
        if data.shape == (ny, nx):
            var = ds.createVariable(
                f"effective_radiance_{channel}", "f4", ("y", "x"),
                zlib=True, fill_value=np.nan,
            )
            var[:] = data
            var.units = "mW m-2 sr-1 (cm-1)-1"
            var.long_name = f"Effective radiance {channel}"
        else:
            # Different resolution channel - store with its own dims
            ch_ny, ch_nx = data.shape
            ds.createDimension(f"y_{channel}", ch_ny)
            ds.createDimension(f"x_{channel}", ch_nx)
            var = ds.createVariable(
                f"effective_radiance_{channel}", "f4",
                (f"y_{channel}", f"x_{channel}"),
                zlib=True, fill_value=np.nan,
            )
            var[:] = data
            var.units = "mW m-2 sr-1 (cm-1)-1"
            var.long_name = f"Effective radiance {channel}"

    # Global attributes
    for key, val in metadata.items():
        ds.setncattr(key, str(val) if not isinstance(val, (int, float)) else val)

    ds.close()
