"""Local cropping of FCI chunks into square patches."""

import os
import numpy as np
import netCDF4 as nc
from pyproj import Proj


# FCI geostationary projection parameters
_FCI_PROJ_PARAMS = dict(
    proj="geos",
    h=35786400.0,
    lon_0=0.0,
    sweep="y",
    ellps="WGS84",
)
_FCI_PROJ = Proj(**_FCI_PROJ_PARAMS)

# Grid sizes per resolution class
_IR_GRID = 5568   # ~2 km
_VIS_GRID = 11136  # ~1 km

# Scale and offset from NetCDF x/y variable attributes (1-based pixel convention)
# x: scale_factor = -5.589e-5, add_offset = +0.15562  (x_rad = col * x_scale + x_offset)
# y: scale_factor = +5.589e-5, add_offset = -0.15562  (y_rad = row * y_scale + y_offset)
_IR_X_SCALE = -5.58871526031607e-05
_IR_X_OFFSET = 0.15561777642350116
_IR_Y_SCALE = 5.58871526031607e-05
_IR_Y_OFFSET = -0.15561777642350116


def _is_vis_channel(channel):
    return channel.startswith("vis_") or channel.startswith("nir_")


def _grid_size(channel):
    return _VIS_GRID if _is_vis_channel(channel) else _IR_GRID


def _get_xy_params(channel):
    """Return (x_scale, x_offset, y_scale, y_offset) for the channel grid."""
    grid = _grid_size(channel)
    if grid == _IR_GRID:
        return _IR_X_SCALE, _IR_X_OFFSET, _IR_Y_SCALE, _IR_Y_OFFSET
    # VIS/NIR: double the grid, half the pixel angular size
    return _IR_X_SCALE / 2, _IR_X_OFFSET, _IR_Y_SCALE / 2, _IR_Y_OFFSET


def latlon_to_fci_pixel(lon, lat, channel):
    """Convert geographic coords to FCI global grid pixel indices.

    Returns (row, col) as float indices into the full-disc grid (1-based).
    """
    x_scale, x_offset, y_scale, y_offset = _get_xy_params(channel)

    # Geographic -> projection coordinates (metres)
    x_m, y_m = _FCI_PROJ(lon, lat)

    # Metres -> radians (angular coordinates)
    x_rad = x_m / _FCI_PROJ_PARAMS["h"]
    y_rad = y_m / _FCI_PROJ_PARAMS["h"]

    # Invert: x_rad = col * x_scale + x_offset  =>  col = (x_rad - x_offset) / x_scale
    #         y_rad = row * y_scale + y_offset  =>  row = (y_rad - y_offset) / y_scale
    col = (x_rad - x_offset) / x_scale
    row = (y_rad - y_offset) / y_scale

    return row, col


def _pixel_range(center_lat, center_lon, size_km, channel):
    """Compute the pixel row/col range for a square patch."""
    grid = _grid_size(channel)
    # Approximate pixel size in km
    pix_km = 2.0 if grid == _IR_GRID else 1.0
    half_pix = int(np.ceil(size_km / 2.0 / pix_km))

    center_row, center_col = latlon_to_fci_pixel(center_lon, center_lat, channel)
    center_row = int(round(center_row))
    center_col = int(round(center_col))

    row_start = max(center_row - half_pix, 0)
    row_end = min(center_row + half_pix, grid)
    col_start = max(center_col - half_pix, 0)
    col_end = min(center_col + half_pix, grid)

    return row_start, row_end, col_start, col_end


def extract_patch(chunk_files, center_lat, center_lon, size_km, channels):
    """Extract a square patch from downloaded FCI chunks.

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
        "row_range" : (start, end)
        "col_range" : (start, end)
    """
    result = {"data": {}}

    for channel in channels:
        row_start, row_end, col_start, col_end = _pixel_range(
            center_lat, center_lon, size_km, channel
        )

        grid = _grid_size(channel)
        patch_rows = row_end - row_start
        patch_cols = col_end - col_start
        patch = np.full((patch_rows, patch_cols), np.nan, dtype=np.float32)

        for cid, fpath in chunk_files.items():
            if cid == "0041":  # Skip trailer chunk
                continue
            try:
                ds = nc.Dataset(fpath)
            except Exception:
                continue

            if channel not in ds["data"].groups:
                ds.close()
                continue

            measured = ds["data"][channel]["measured"]
            chunk_row_start = int(measured.variables["start_position_row"][:])
            chunk_row_end = int(measured.variables["end_position_row"][:])
            chunk_col_start = int(measured.variables["start_position_column"][:])
            chunk_col_end = int(measured.variables["end_position_column"][:])

            # Convert to 0-based
            chunk_row_start -= 1
            chunk_row_end -= 1
            chunk_col_start -= 1
            chunk_col_end -= 1

            # Overlap between patch and chunk
            ov_row_start = max(row_start, chunk_row_start)
            ov_row_end = min(row_end, chunk_row_end + 1)
            ov_col_start = max(col_start, chunk_col_start)
            ov_col_end = min(col_end, chunk_col_end + 1)

            if ov_row_start >= ov_row_end or ov_col_start >= ov_col_end:
                ds.close()
                continue

            # Local indices into the chunk array
            local_row_start = ov_row_start - chunk_row_start
            local_row_end = ov_row_end - chunk_row_start
            local_col_start = ov_col_start - chunk_col_start
            local_col_end = ov_col_end - chunk_col_start

            # Indices into the patch array
            patch_row_start = ov_row_start - row_start
            patch_row_end = ov_row_end - row_start
            patch_col_start = ov_col_start - col_start
            patch_col_end = ov_col_end - col_start

            radiance = measured.variables["effective_radiance"]
            scale = getattr(radiance, "scale_factor", 1.0)
            offset = getattr(radiance, "add_offset", 0.0)

            raw = radiance[local_row_start:local_row_end, local_col_start:local_col_end]
            patch[patch_row_start:patch_row_end, patch_col_start:patch_col_end] = (
                np.array(raw, dtype=np.float32) * scale + offset
            )

            ds.close()

        result["data"][channel] = patch

    # Store pixel ranges (use first channel for geolocation reference)
    ref_channel = channels[0]
    row_start, row_end, col_start, col_end = _pixel_range(
        center_lat, center_lon, size_km, ref_channel
    )
    result["row_range"] = (row_start, row_end)
    result["col_range"] = (col_start, col_end)

    # Build geolocation arrays from the reference channel grid
    lat_2d, lon_2d = build_geolocation(row_start, row_end, col_start, col_end, ref_channel)
    result["lat"] = lat_2d
    result["lon"] = lon_2d

    return result


def build_geolocation(row_start, row_end, col_start, col_end, channel):
    """Inverse geostationary projection: pixel indices -> 2D lat/lon arrays."""
    x_scale, x_offset, y_scale, y_offset = _get_xy_params(channel)

    rows = np.arange(row_start, row_end)
    cols = np.arange(col_start, col_end)

    # Pixel -> angular coordinates (radians)
    # x_rad = col * x_scale + x_offset
    # y_rad = row * y_scale + y_offset
    x_rad = cols * x_scale + x_offset
    y_rad = rows * y_scale + y_offset

    xx, yy = np.meshgrid(x_rad, y_rad)

    # Angular -> metres
    xx_m = xx * _FCI_PROJ_PARAMS["h"]
    yy_m = yy * _FCI_PROJ_PARAMS["h"]

    # Inverse projection -> lon, lat
    lon_2d, lat_2d = _FCI_PROJ(xx_m, yy_m, inverse=True)

    # Mask points outside the disc (pyproj returns 1e30 or inf)
    invalid = (np.abs(lon_2d) > 180) | (np.abs(lat_2d) > 90)
    lon_2d[invalid] = np.nan
    lat_2d[invalid] = np.nan

    return lat_2d.astype(np.float32), lon_2d.astype(np.float32)


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
