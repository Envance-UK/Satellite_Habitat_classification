"""
Step 5 — Compute per-pixel temporal statistics from the data cube.

For every band in the cube, computes across the time dimension:
    mean, median, min, max, std, range

Output is a flat multi-band GeoTIFF:
    17 bands x 6 statistics = 102 bands total

Band order:
    Bands 1-6:   B2    (mean, median, min, max, std, range)
    Bands 7-12:  B3    (mean, median, min, max, std, range)
    ...and so on for B4, B5, B6, B7, B8, B11, B12,
    NDVI, NDWI, NBR, NDRE, SAVI, EVI, NDMI, BSI

Reads paths from Project/Chelmsford_phase_2_config.py.
"""

import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import xarray as xr

# Load config
_config_path = Path(__file__).parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

STAT_NAMES = ["mean", "median", "min", "max", "std", "range"]


def compute_stats(arr):
    """
    Compute temporal statistics for a (time, y, x) array.
    NaN values are excluded from all calculations.
    Returns a list of (stat_name, 2D array) tuples.
    """
    return [
        ("mean",   np.nanmean(arr,   axis=0).astype("float32")),
        ("median", np.nanmedian(arr, axis=0).astype("float32")),
        ("min",    np.nanmin(arr,    axis=0).astype("float32")),
        ("max",    np.nanmax(arr,    axis=0).astype("float32")),
        ("std",    np.nanstd(arr,    axis=0).astype("float32")),
        ("range",  (np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)).astype("float32")),
    ]


def main():
    input_path  = Path(cfg.STATS_INPUT)
    output_path = Path(cfg.STATS_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data cube: {input_path.name}")
    ds = xr.open_dataset(input_path)
    da = ds["reflectance"]  # (time, band, y, x)

    band_names = da.coords["band"].values.tolist()
    y_coords   = da.coords["y"].values
    x_coords   = da.coords["x"].values
    n_bands    = len(band_names)

    # Reconstruct affine transform from coordinate arrays
    x_res = float(x_coords[1] - x_coords[0])
    y_res = float(y_coords[1] - y_coords[0])
    transform = rasterio.transform.from_origin(
        west  = float(x_coords[0]) - x_res / 2,
        north = float(y_coords[0]) - y_res / 2,
        xsize = abs(x_res),
        ysize = abs(y_res),
    )

    # Pull CRS from dataset attributes
    crs = da.attrs.get("crs", "EPSG:32630")

    total_output_bands = n_bands * len(STAT_NAMES)
    height = len(y_coords)
    width  = len(x_coords)

    print(f"  {len(da.coords['time'])} time steps | {n_bands} bands | "
          f"output: {total_output_bands} bands ({n_bands} x {len(STAT_NAMES)} stats)")

    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     width,
        "height":    height,
        "count":     total_output_bands,
        "crs":       crs,
        "transform": transform,
        "nodata":    np.nan,
    }

    band_descriptions = []
    out_band_idx = 1

    with rasterio.open(output_path, "w", **profile) as dst:
        for b_idx, b_name in enumerate(band_names):
            print(f"  Computing stats for {b_name} ...")
            arr = da.isel(band=b_idx).values.astype("float32")  # (time, y, x)

            for stat_name, stat_arr in compute_stats(arr):
                dst.write(stat_arr, out_band_idx)
                desc = f"{b_name}_{stat_name}"
                dst.update_tags(out_band_idx, name=desc)
                band_descriptions.append(desc)
                out_band_idx += 1

    print(f"\nStatistics GeoTIFF saved → {output_path}")
    print(f"  Total bands: {total_output_bands}")
    print(f"  Band order:  {', '.join(band_descriptions[:6])} ... {band_descriptions[-1]}")


if __name__ == "__main__":
    main()
