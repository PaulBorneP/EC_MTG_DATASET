"""MTG FCI chunk selection and download via EUMETSAT Data Store."""

import os
import re
import shutil
import datetime
import dotenv
import eumdac


def authenticate():
    """Create an EUMETSAT Data Store access token from .env credentials."""
    dotenv.load_dotenv()
    key = os.getenv("EUMETSAT_KEY")
    secret = os.getenv("EUMETSAT_SECRET")
    if not key or not secret:
        raise RuntimeError("EUMETSAT_KEY and EUMETSAT_SECRET must be set in .env")
    return eumdac.AccessToken((key, secret))


def find_mtg_products(token, collection_id, ec_time_start, ec_time_end, margin_minutes=10):
    """Search Data Store for FCI products overlapping the EC overpass window.

    Parameters
    ----------
    token : eumdac.AccessToken
    collection_id : str  (e.g. "EO:EUM:DAT:0662")
    ec_time_start, ec_time_end : datetime.datetime
        Start/end of the EarthCARE overpass window.
    margin_minutes : int
        Extra margin on each side of the time window.

    Returns
    -------
    list of eumdac products
    """
    datastore = eumdac.DataStore(token)
    collection = datastore.get_collection(collection_id)
    margin = datetime.timedelta(minutes=margin_minutes)
    products = collection.search(
        dtstart=ec_time_start - margin,
        dtend=ec_time_end + margin,
    )
    return list(products)


def get_product_time_range(product):
    """Extract sensing start/end times from a product ID string.

    MTG product IDs contain OPE_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.
    Returns (start_dt, end_dt).
    """
    pid = str(product)
    match = re.search(r"OPE_(\d{14})_(\d{14})", pid)
    if not match:
        return None, None
    fmt = "%Y%m%d%H%M%S"
    return (
        datetime.datetime.strptime(match.group(1), fmt),
        datetime.datetime.strptime(match.group(2), fmt),
    )


def get_chunk_scan_time(chunk_filename):
    """Parse scan start/end times from a chunk filename.

    Chunk filenames contain OPE_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.
    Returns (start_dt, end_dt) or (None, None).
    """
    match = re.search(r"OPE_(\d{14})_(\d{14})", chunk_filename)
    if not match:
        return None, None
    fmt = "%Y%m%d%H%M%S"
    return (
        datetime.datetime.strptime(match.group(1), fmt),
        datetime.datetime.strptime(match.group(2), fmt),
    )


def get_chunk_id(entry_name):
    """Extract the 4-digit chunk ID from a product entry filename.

    Chunk files end with _RRRR_CCCC.nc where CCCC is the chunk number.
    """
    match = re.search(r"_(\d{4})\.nc$", entry_name)
    return match.group(1) if match else None


def download_chunks(product, chunk_ids, output_dir):
    """Download specific chunk NetCDF files from a product.

    Parameters
    ----------
    product : eumdac product object
    chunk_ids : list of str  (e.g. ["0021", "0022", "0041"])
    output_dir : str
        Directory to save downloaded files.

    Returns
    -------
    dict mapping chunk_id -> local file path
    """
    os.makedirs(output_dir, exist_ok=True)
    # Always include trailer chunk
    all_ids = set(chunk_ids) | {"0041"}

    # Build suffix patterns for matching
    patterns = [f"_{cid}.nc" for cid in all_ids]

    downloaded = {}
    for entry in product.entries:
        entry_str = str(entry)
        if not any(entry_str.endswith(p) for p in patterns):
            continue
        cid = get_chunk_id(entry_str)
        if cid is None:
            continue

        local_path = os.path.join(output_dir, os.path.basename(entry_str))
        if os.path.exists(local_path):
            # Verify existing cached file is not truncated
            try:
                import netCDF4 as _nc4
                with _nc4.Dataset(local_path, "r"):
                    pass
                downloaded[cid] = local_path
                continue
            except Exception:
                print(f"  Cached chunk appears corrupt, re-downloading: {os.path.basename(local_path)}")
                os.remove(local_path)

        for attempt in range(2):
            try:
                with product.open(entry=entry_str) as fsrc:
                    with open(local_path, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                # Verify the downloaded file is not truncated
                import netCDF4 as _nc4
                with _nc4.Dataset(local_path, "r"):
                    pass
                downloaded[cid] = local_path
                break
            except Exception as e:
                if os.path.exists(local_path):
                    os.remove(local_path)
                if attempt == 0:
                    print(f"  Warning: download failed or truncated for {entry_str}, retrying: {e}")
                else:
                    print(f"  Warning: download failed after retry for {entry_str}: {e}")

    return downloaded
