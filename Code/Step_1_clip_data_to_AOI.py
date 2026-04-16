"""
Step 1 — Clip Sentinel-2 images to AOI.

Reads paths from Project/Chelmsford_phase_2_config.py.
"""

import importlib.util
from pathlib import Path

# Load config directly from its file path — avoids any sys.path / package issues
_config_path = Path(__file__).parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

import fiona
import rasterio
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely.geometry import shape, mapping
from shapely.ops import transform
import pyproj


def load_aoi_shapes(aoi_path):
    with fiona.open(aoi_path) as src:
        aoi_crs = src.crs
        geoms = [feature["geometry"] for feature in src]
    return geoms, aoi_crs


def reproject_geoms(geoms, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    ).transform
    return [mapping(transform(project, shape(g))) for g in geoms]


def clip_image(input_path, output_path, aoi_path):
    with rasterio.open(input_path) as src:
        aoi_geoms, aoi_crs = load_aoi_shapes(aoi_path)
        raster_crs = src.crs

        if CRS.from_user_input(aoi_crs) != raster_crs:
            aoi_geoms = reproject_geoms(aoi_geoms, aoi_crs, raster_crs)
        else:
            aoi_geoms = [mapping(shape(g)) for g in aoi_geoms]

        clipped, transform_ = mask(src, aoi_geoms, crop=True)
        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform_,
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(clipped)


def main():
    input_dir  = Path(cfg.INPUT_DIR)
    output_dir = Path(cfg.OUTPUT_DIR)
    aoi_path   = cfg.AOI

    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not images:
        print("No .tif / .tiff files found in input directory.")
        return

    for img_path in sorted(images):
        out_path = output_dir / img_path.name
        print(f"Clipping {img_path.name} ...", end=" ", flush=True)
        try:
            clip_image(img_path, out_path, aoi_path)
            print("done")
        except Exception as e:
            print(f"FAILED — {e}")

    print(f"\nFinished. {len(images)} image(s) processed → {output_dir}")


if __name__ == "__main__":
    main()
