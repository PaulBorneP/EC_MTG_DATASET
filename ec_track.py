"""EarthCARE track extraction and patch geometry."""

import h5py
import numpy as np
from datetime import datetime, timedelta
from pyproj import Geod
from shapely.geometry import Polygon, box
from shapely.wkt import loads


# Reference epoch for EC time field
_EC_EPOCH = datetime(2000, 1, 1)

# WGS84 ellipsoid for distance calculations
_GEOD = Geod(ellps="WGS84")

# MTG subsatellite point
_MTG_LON = 0.0


def load_ec_track(filepath):
    """Read AC_TC_2B HDF5 and return lat, lon, time arrays.

    Returns
    -------
    lats : np.ndarray (N,)
    lons : np.ndarray (N,)
    times : np.ndarray of datetime (N,)
    """
    with h5py.File(filepath, "r") as f:
        lats = f["ScienceData/latitude"][:]
        lons = f["ScienceData/longitude"][:]
        secs = f["ScienceData/time"][:]

    times = np.array([_EC_EPOCH + timedelta(seconds=float(s)) for s in secs])
    return lats, lons, times


def filter_to_mtg_disc(lats, lons, max_zenith_deg=75.0):
    """Keep only points visible from MTG geostationary position.

    Approximates viewing zenith angle check by limiting angular distance
    from the subsatellite point to max_zenith_deg in longitude and latitude.
    """
    # Simple geometric disc approximation
    lon_diff = np.abs(lons - _MTG_LON)
    lon_diff = np.where(lon_diff > 180, 360 - lon_diff, lon_diff)
    angular_dist = np.sqrt(lats**2 + lon_diff**2)
    mask = angular_dist <= max_zenith_deg
    return mask


def generate_patch_centers(lats, lons, times, spacing_km):
    """Subsample track at regular along-track intervals.

    Returns arrays of (center_lat, center_lon, center_time) spaced
    approximately spacing_km apart along the ground track.
    """
    centers_idx = [0]
    cumulative = 0.0

    for i in range(1, len(lats)):
        _, _, dist = _GEOD.inv(lons[i - 1], lats[i - 1], lons[i], lats[i])
        cumulative += dist / 1000.0  # m -> km
        if cumulative >= spacing_km:
            centers_idx.append(i)
            cumulative = 0.0

    idx = np.array(centers_idx)
    return lats[idx], lons[idx], times[idx]


def compute_patch_bbox(lat, lon, size_km):
    """Compute a square bounding box around (lat, lon).

    Returns (min_lon, min_lat, max_lon, max_lat).
    """
    half = size_km / 2.0 * 1000.0  # km -> m

    # North, East, South, West azimuths
    lon_e, lat_n, _ = _GEOD.fwd(lon, lat, 0, half)
    lon_e, _, _ = _GEOD.fwd(lon, lat, 90, half)
    _, lat_s, _ = _GEOD.fwd(lon, lat, 180, half)
    lon_w, _, _ = _GEOD.fwd(lon, lat, 270, half)

    return (lon_w, lat_s, lon_e, lat_n)


def load_chunk_polygons(wkt_file):
    """Load FCI chunk footprints from WKT file.

    Returns dict of {chunk_id_str: shapely.Polygon}.
    """
    polygons = {}
    with open(wkt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk_id, wkt_poly = line.split(",", 1)
            polygons[chunk_id] = loads(wkt_poly)
    return polygons


def find_intersecting_chunks(bbox, chunk_polygons):
    """Return list of chunk IDs whose footprints intersect the bbox.

    Parameters
    ----------
    bbox : tuple (min_lon, min_lat, max_lon, max_lat)
    chunk_polygons : dict from load_chunk_polygons()
    """
    patch_box = box(*bbox)
    return [cid for cid, poly in chunk_polygons.items() if patch_box.intersects(poly)]
