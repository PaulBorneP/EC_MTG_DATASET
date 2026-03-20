"""Orchestration script: EarthCARE ↔ MTG FCI collocation pipeline."""

import faulthandler
faulthandler.enable()

import os
import csv
import glob
import yaml
from datetime import timedelta
from tqdm import tqdm

from ec_track import (
    load_ec_track,
    filter_to_mtg_disc,
    generate_patch_centers,
    compute_patch_bbox,
    load_chunk_polygons,
    find_intersecting_chunks,
)
from mtg_download import (
    authenticate,
    find_mtg_products,
    get_product_time_range,
    get_chunk_scan_time,
    download_chunks,
)
from patch_extractor import extract_patch, save_patch


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _ec_orbit_id(filepath):
    """Extract orbit ID from EC filename (e.g. '07470A')."""
    basename = os.path.basename(filepath)
    # Pattern: ...Z_ORBITID.h5
    parts = basename.replace(".h5", "").split("_")
    return parts[-1] if parts else basename


def run(config_path="config.yaml"):
    cfg = load_config(config_path)
    patch_size_km = cfg["patch_size_km"]
    channels = cfg["fci_channels"]
    collection_id = cfg["fci_collection"]
    max_dt = cfg["max_time_diff_minutes"] * 60  # seconds
    ec_dir = cfg["ec_data_dir"]
    mtg_dir = cfg["mtg_data_dir"]
    patch_dir = cfg["patch_output_dir"]
    wkt_file = cfg["chunk_wkt_file"]
    dataset_bbox = cfg["dataset_bbox"]

    os.makedirs(patch_dir, exist_ok=True)

    # Load chunk geometry once
    chunk_polygons = load_chunk_polygons(wkt_file)

    # Authenticate with EUMETSAT
    print("Authenticating with EUMETSAT Data Store...")
    token = authenticate()

    # Gather EC files
    ec_files = sorted(glob.glob(os.path.join(ec_dir, "*.h5")))
    if not ec_files:
        print(f"No EC files found in {ec_dir}")
        return

    print(f"Found {len(ec_files)} EarthCARE files")

    # Index CSV
    index_path = os.path.join(patch_dir, "index.csv")
    index_rows = []

    for ec_file in tqdm(ec_files, desc="Processing EC files"):
        orbit_id = _ec_orbit_id(ec_file)
        print(f"\n--- Processing {os.path.basename(ec_file)} (orbit {orbit_id}) ---")

        # Step 1: Load track
        lats, lons, times = load_ec_track(ec_file)
        print(f"  Track: {len(lats)} points, lat [{lats.min():.1f}, {lats.max():.1f}], "
              f"lon [{lons.min():.1f}, {lons.max():.1f}]")

        # Step 2: Filter to MTG disc
        #filter lats/lons to dataset_bbox first to speed up disc filtering
        bbox_mask = (
            (lats >= dataset_bbox[0]) & (lats <= dataset_bbox[1]) &
            (lons >= dataset_bbox[2]) & (lons <= dataset_bbox[3])
        )
        lats, lons, times = lats[bbox_mask], lons[bbox_mask], times[bbox_mask]
        disc_mask = filter_to_mtg_disc(lats, lons)
        lats, lons, times = lats[disc_mask], lons[disc_mask], times[disc_mask]
        if len(lats) == 0:
            print("  No points within MTG disc, skipping")
            continue
        print(f"  After disc filter: {len(lats)} points")

        # Step 3: Generate patch centers
        c_lats, c_lons, c_times = generate_patch_centers(
            lats, lons, times, patch_size_km
        )
        print(f"  Patch centers: {len(c_lats)}")

        if len(c_lats) == 0:
            continue

        # Step 4: For each patch, find needed chunks
        patches_info = []
        all_needed_chunks = set()

        for i in range(len(c_lats)):
            bbox = compute_patch_bbox(c_lats[i], c_lons[i], patch_size_km)
            chunks = find_intersecting_chunks(bbox, chunk_polygons)
            patches_info.append({
                "idx": i,
                "lat": float(c_lats[i]),
                "lon": float(c_lons[i]),
                "time": c_times[i],
                "bbox": bbox,
                "chunks": chunks,
            })
            all_needed_chunks.update(chunks)

        print(f"  Unique chunks needed: {sorted(all_needed_chunks)}")

        # Step 5: Find matching MTG products
        ec_time_start = min(c_times)
        ec_time_end = max(c_times)
        products = find_mtg_products(
            token, collection_id, ec_time_start, ec_time_end
        )
        if not products:
            print("  No MTG products found for this time window, skipping")
            continue
        print(f"  Found {len(products)} MTG product(s)")

        # Step 6: Download chunks from each product
        # Pick the product closest in time
        best_product = None
        best_dt = float("inf")
        ec_mid = ec_time_start + (ec_time_end - ec_time_start) / 2

        for prod in products:
            t_start, t_end = get_product_time_range(prod)
            if t_start is None:
                continue
            prod_mid = t_start + (t_end - t_start) / 2
            dt = abs((prod_mid - ec_mid).total_seconds())
            if dt < best_dt:
                best_dt = dt
                best_product = prod

        if best_product is None:
            print("  Could not determine product timing, skipping")
            continue

        print(f"  Best product: {best_product}")
        print(f"  Downloading {len(all_needed_chunks)} chunks...")
        chunk_files = download_chunks(
            best_product, list(all_needed_chunks), mtg_dir
        )
        print(f"  Downloaded/cached {len(chunk_files)} chunk files")

        # Step 7: Extract and save patches
        for pinfo in patches_info:
            # Get the chunk scan midpoint time for time-diff calculation
            scan_times = []
            for cid in pinfo["chunks"]:
                if cid in chunk_files:
                    t_s, t_e = get_chunk_scan_time(
                        os.path.basename(chunk_files[cid])
                    )
                    if t_s:
                        scan_times.append(t_s + (t_e - t_s) / 2)

            if scan_times:
                mtg_time = min(scan_times, key=lambda t: abs(
                    (t - pinfo["time"]).total_seconds()
                ))
                time_diff_s = abs((mtg_time - pinfo["time"]).total_seconds())
            else:
                mtg_time = None
                time_diff_s = float("inf")

            if time_diff_s > max_dt:
                print(f"  Patch {pinfo['idx']:03d}: time diff {time_diff_s:.0f}s > "
                      f"{max_dt}s, skipping")
                continue

            # Extract patch
            try:
                patch_data = extract_patch(
                    chunk_files, pinfo["lat"], pinfo["lon"],
                    patch_size_km, channels,
                )
            except Exception as e:
                print(f"  Patch {pinfo['idx']:03d}: extraction failed, skipping: {e}")
                continue

            # Build output filename
            ec_time_str = pinfo["time"].strftime("%Y%m%dT%H%M%S")
            patch_fname = f"EC_MTG_patch_{orbit_id}_{pinfo['idx']:03d}_{ec_time_str}.nc"
            patch_path = os.path.join(patch_dir, patch_fname)

            metadata = {
                "ec_source_file": os.path.basename(ec_file),
                "mtg_product_id": str(best_product),
                "center_lat": pinfo["lat"],
                "center_lon": pinfo["lon"],
                "patch_size_km": patch_size_km,
                "ec_time": str(pinfo["time"]),
                "mtg_time": str(mtg_time) if mtg_time else "unknown",
                "time_diff_seconds": time_diff_s,
                "channels": ",".join(channels),
            }

            save_patch(patch_data, metadata, patch_path)

            index_rows.append({
                "ec_file": os.path.basename(ec_file),
                "patch_idx": pinfo["idx"],
                "center_lat": pinfo["lat"],
                "center_lon": pinfo["lon"],
                "ec_time": str(pinfo["time"]),
                "mtg_time": str(mtg_time) if mtg_time else "",
                "time_diff_s": time_diff_s,
                "mtg_chunks": ";".join(pinfo["chunks"]),
                "patch_file": patch_fname,
            })

            print(f"  Patch {pinfo['idx']:03d}: saved ({time_diff_s:.0f}s diff)")

    # Write index CSV
    if index_rows:
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=index_rows[0].keys())
            writer.writeheader()
            writer.writerows(index_rows)
        print(f"\nWrote {len(index_rows)} patches to {index_path}")
    else:
        print("\nNo patches generated")


if __name__ == "__main__":
    run()
