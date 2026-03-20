#!/usr/bin/env python
"""Compute global p2/p98 percentiles per band across all extracted patches.

Writes band_stats.json with entries for each individual channel and for
difference bands used in RGB recipes (e.g. ir_123-ir_105 for the fog composite).
"""

import json
import glob
import numpy as np
import xarray as xr

PATCH_DIR = "DATA/patches"
OUTPUT_FILE = "band_stats.json"
SAMPLES_PER_PATCH = 10_000

# All individual channels stored in patch files
CHANNELS = [
    "ir_105", "ir_123", "ir_87", "ir_38", "ir_97", "ir_133",
    "wv_63", "wv_73",
    "vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
    "nir_13", "nir_16", "nir_22",
]

# Difference bands used in RGB recipes (ch1 - ch2)
DIFF_BANDS = [
    ("ir_123", "ir_105"),
    ("ir_105", "ir_38"),
]


def _subsample(arr, n):
    """Return up to n random non-NaN values from a 2-D array."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return valid
    if len(valid) <= n:
        return valid
    return np.random.default_rng(42).choice(valid, size=n, replace=False)


def _load_diff_band(ds, ch1, ch2):
    """Load a difference band (ch1 - ch2), matching resolution if needed."""
    v1 = ds[f"effective_radiance_{ch1}"].values.astype(np.float32)
    v2 = ds[f"effective_radiance_{ch2}"].values.astype(np.float32)
    if v1.shape != v2.shape:
        v2 = np.repeat(
            np.repeat(v2, v1.shape[0] // v2.shape[0], axis=0),
            v1.shape[1] // v2.shape[1], axis=1,
        )
    return v1 - v2


def main():
    patch_files = sorted(glob.glob(f"{PATCH_DIR}/EC_MTG_patch_*.nc"))
    if not patch_files:
        print(f"No patch files found in {PATCH_DIR}/")
        return

    print(f"Found {len(patch_files)} patch files")

    # Collect subsampled values per band spec
    samples = {ch: [] for ch in CHANNELS}
    for ch1, ch2 in DIFF_BANDS:
        samples[f"{ch1}-{ch2}"] = []

    for i, path in enumerate(patch_files):
        print(f"  [{i+1}/{len(patch_files)}] {path}")
        ds = xr.open_dataset(path, engine="h5netcdf")

        for ch in CHANNELS:
            var = f"effective_radiance_{ch}"
            if var in ds.data_vars:
                arr = ds[var].values.astype(np.float32)
                samples[ch].append(_subsample(arr, SAMPLES_PER_PATCH))

        for ch1, ch2 in DIFF_BANDS:
            v1 = f"effective_radiance_{ch1}"
            v2 = f"effective_radiance_{ch2}"
            if v1 in ds.data_vars and v2 in ds.data_vars:
                diff = _load_diff_band(ds, ch1, ch2)
                samples[f"{ch1}-{ch2}"].append(_subsample(diff, SAMPLES_PER_PATCH))

        ds.close()

    # Compute percentiles
    stats = {}
    for key, vals in samples.items():
        if not vals:
            print(f"  WARNING: no data for {key}, skipping")
            continue
        combined = np.concatenate(vals)
        p2, p98 = np.nanpercentile(combined, [2, 98])
        stats[key] = {"p2": float(p2), "p98": float(p98)}
        print(f"  {key}: p2={p2:.4f}  p98={p98:.4f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nWrote {len(stats)} band stats to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
