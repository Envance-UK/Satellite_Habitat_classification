"""
Step 2 — Apply SCL cloud mask to Sentinel-2 images.

For each image:
  - Band 13 (SCL) is used as the classification mask
  - Pixels in bands 1-12 where SCL is in {3, 8, 9, 10} are set to NoData
  - Output is exported without band 13

SCL classes masked:
  3  = Cloud shadow
  8  = Cloud (medium probability)
  9  = Cloud (high probability)
  10 = Thin cirrus

Reads paths from Project/Chelmsford_phase_2_config.py.
"""

import importlib.util
from pathlib import Path

import numpy as np
import rasterio

# Load config
_config_path = Path(__file__).parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

MASK_VALUES = {3, 8, 9, 10}


def process_image(input_path, output_path):
    with rasterio.open(input_path) as src:
        if src.count != 13:
            print(f"  WARNING: expected 13 bands, got {src.count} — skipping")
            return

        # Read SCL (band 13) and determine nodata
        scl = src.read(13)

        # Build cloud mask: True where pixel should be masked
        cloud_mask = np.isin(scl, list(MASK_VALUES))

        profile = src.profile.copy()

        if src.nodata is not None:
            # Use existing nodata value, keep original dtype
            nodata = src.nodata
            profile.update(count=12, nodata=nodata)

            with rasterio.open(output_path, "w", **profile) as dst:
                for band_idx in range(1, 13):
                    data = src.read(band_idx)
                    data[cloud_mask] = nodata
                    dst.write(data, band_idx)
        else:
            # No nodata defined — convert to float32 so NaN can be used
            profile.update(count=12, nodata=np.nan, dtype="float32")

            with rasterio.open(output_path, "w", **profile) as dst:
                for band_idx in range(1, 13):
                    data = src.read(band_idx).astype("float32")
                    data[cloud_mask] = np.nan
                    dst.write(data, band_idx)


def main():
    input_dir  = Path(cfg.CLOUD_MASK_INPUT)
    output_dir = Path(cfg.CLOUD_MASK_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not images:
        print("No .tif / .tiff files found in input directory.")
        return

    for img_path in sorted(images):
        out_path = output_dir / img_path.name
        print(f"Masking {img_path.name} ...", end=" ", flush=True)
        try:
            process_image(img_path, out_path)
            print("done")
        except Exception as e:
            print(f"FAILED — {e}")

    print(f"\nFinished. {len(images)} image(s) processed → {output_dir}")


if __name__ == "__main__":
    main()
