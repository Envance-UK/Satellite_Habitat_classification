"""
Step 4 — Build a NetCDF data cube from all processed Sentinel-2 images.

Reads every .tif from DATA_CUBE_INPUT and stacks them into a 4-D xarray DataArray:
    dimensions: (time, band, y, x)

Band names reflect the 17-band output from Step 3:
    B2, B3, B4, B5, B6, B7, B8, B11, B12,
    NDVI, NDWI, NBR, NDRE, SAVI, EVI, NDMI, BSI

Time is parsed from the Sentinel-2 filename (YYYYMMDDTHHMMSS).
Spatial coordinates (x, y) are derived from the raster transform.

Reads paths from Project/Chelmsford_phase_2_config.py.
"""

import importlib.util
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import xarray as xr
from datetime import datetime

# Load config
_config_path = Path(__file__).parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

# Band names matching Step 3 output (17 bands)
BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12",
    "NDVI", "NDWI", "NBR", "NDRE", "SAVI", "EVI", "NDMI", "BSI",
]


def parse_date(filename):
    """Extract acquisition datetime from a Sentinel-2 filename."""
    match = re.search(r"(\d{8}T\d{6})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    raise ValueError(f"Could not parse date from filename: {filename}")


def read_and_align(image_path, ref_shape, ref_transform, ref_crs):
    """Read a GeoTIFF, resampling to match the reference grid if needed."""
    with rasterio.open(image_path) as src:
        if (src.height, src.width) == (ref_shape[0], ref_shape[1]):
            return src.read().astype("float32")

        print(f"  Resampling {image_path.name} to match reference grid...")
        n_bands = src.count
        data = np.zeros((n_bands, ref_shape[0], ref_shape[1]), dtype="float32")
        for i in range(1, n_bands + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=data[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
        return data


def build_xy_coords(transform, height, width):
    """Compute x and y coordinate arrays from a rasterio Affine transform."""
    cols = np.arange(width)
    rows = np.arange(height)
    x_coords = transform.c + (cols + 0.5) * transform.a
    y_coords = transform.f + (rows + 0.5) * transform.e
    return x_coords, y_coords


def create_data_cube(image_paths, output_path):
    # Sort images chronologically
    image_paths = sorted(image_paths, key=lambda p: parse_date(p.name))

    # Reference grid from the first image
    with rasterio.open(image_paths[0]) as ref:
        ref_shape  = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs    = ref.crs
        ref_nodata = ref.nodata
        x_coords, y_coords = build_xy_coords(ref_transform, ref.height, ref.width)

    # Load all images
    print(f"Loading {len(image_paths)} image(s)...")
    arrays = []
    times  = []
    for path in image_paths:
        print(f"  {path.name}")
        data = read_and_align(path, ref_shape, ref_transform, ref_crs)
        arrays.append(data)
        times.append(parse_date(path.name))

    # Stack → (time, band, y, x)
    cube = np.stack(arrays, axis=0)

    # Determine band names — use defaults if band count doesn't match
    n_bands = cube.shape[1]
    if n_bands == len(BAND_NAMES):
        band_names = BAND_NAMES
    else:
        print(f"  WARNING: expected {len(BAND_NAMES)} bands, got {n_bands}. Using generic names.")
        band_names = [f"Band{i + 1}" for i in range(n_bands)]

    # Build xarray DataArray
    da = xr.DataArray(
        cube,
        dims=["time", "band", "y", "x"],
        coords={
            "time": times,
            "band": band_names,
            "y":    y_coords,
            "x":    x_coords,
        },
        attrs={
            "crs":       ref_crs.to_string(),
            "transform": list(ref_transform),
            "nodata":    ref_nodata if ref_nodata is not None else "NaN",
            "description": "Chelmsford Phase 2 — Sentinel-2 multitemporal data cube",
        },
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = da.to_dataset(name="reflectance")
    ds.to_netcdf(output_path)
    print(f"\nData cube saved → {output_path}")
    print(f"  Shape: {dict(zip(da.dims, da.shape))}")


def main():
    input_dir   = Path(cfg.DATA_CUBE_INPUT)
    output_path = Path(cfg.DATA_CUBE_OUTPUT)

    images = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not images:
        print("No .tif / .tiff files found in input directory.")
        return

    create_data_cube(images, output_path)


if __name__ == "__main__":
    main()
