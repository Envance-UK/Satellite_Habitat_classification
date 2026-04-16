"""
Step 3 — Compute spectral indices, normalise original bands, remove unwanted bands.

Input bands (12-band cloud-masked image from Step 2):
    1: B1  - Coastal aerosol (443nm)
    2: B2  - Blue (490nm)
    3: B3  - Green (560nm)
    4: B4  - Red (665nm)
    5: B5  - Red Edge 1 (705nm)
    6: B6  - Red Edge 2 (740nm)
    7: B7  - Red Edge 3 (783nm)
    8: B8  - NIR (842nm)
    9: B8A - Narrow NIR (865nm)
   10: B9  - Water vapour (945nm)
   11: B11 - SWIR 1 (1610nm)
   12: B12 - SWIR 2 (2190nm)

Processing order:
    1. Compute indices → appended as bands 13-20
    2. Normalise original bands 2-8, 11, 12 (kept reflectance bands)
    3. Remove original bands 1, 9, 10 (aerosol, narrow NIR, water vapour)

Output band order:
    1:  B2  - Blue           (normalised)
    2:  B3  - Green          (normalised)
    3:  B4  - Red            (normalised)
    4:  B5  - Red Edge 1     (normalised)
    5:  B6  - Red Edge 2     (normalised)
    6:  B7  - Red Edge 3     (normalised)
    7:  B8  - NIR            (normalised)
    8:  B11 - SWIR 1         (normalised)
    9:  B12 - SWIR 2         (normalised)
   10:  NDVI
   11:  NDWI
   12:  NBR
   13:  NDRE
   14:  SAVI
   15:  EVI
   16:  NDMI
   17:  BSI

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

# Bands (1-based) to remove from the original 12
BANDS_TO_REMOVE = {1, 9, 10}  # B1 (aerosol), B8A (narrow NIR), B9 (water vapour)


def nodata_mask(arr, nodata):
    """True where pixel is nodata (handles both NaN and numeric nodata)."""
    if nodata is not None and np.isnan(nodata):
        return np.isnan(arr)
    elif nodata is not None:
        return arr == nodata
    else:
        return np.zeros(arr.shape, dtype=bool)


def safe_ratio(num, den, mask, fill=0.0):
    """(num - den) / (num + den) with epsilon to avoid divide-by-zero."""
    result = np.where(
        np.abs(num + den) > 1e-10,
        num / (den + 1e-10),
        fill,
    )
    result[mask] = fill
    return result.astype(np.float32)


def compute_indices(src, nodata):
    """Compute 8 spectral indices from the 12-band image."""
    # Read required bands as float
    b2  = src.read(2).astype(np.float32)   # Blue
    b3  = src.read(3).astype(np.float32)   # Green
    b4  = src.read(4).astype(np.float32)   # Red
    b5  = src.read(5).astype(np.float32)   # Red Edge 1
    b8  = src.read(8).astype(np.float32)   # NIR
    b8a = src.read(9).astype(np.float32)   # Narrow NIR (B8A)
    b11 = src.read(11).astype(np.float32)  # SWIR 1
    b12 = src.read(12).astype(np.float32)  # SWIR 2

    # Nodata mask — any pixel that is nodata in any used band
    nd = nodata_mask
    mask = (
        nd(b2, nodata) | nd(b3, nodata) | nd(b4, nodata) |
        nd(b5, nodata) | nd(b8, nodata) | nd(b8a, nodata) |
        nd(b11, nodata) | nd(b12, nodata)
    )

    fill = np.nan if (nodata is None or (nodata is not None and np.isnan(float(nodata)))) else nodata

    # NDVI — Normalised Difference Vegetation Index
    # (NIR - Red) / (NIR + Red)
    ndvi = safe_ratio(b8 - b4, b8 + b4, mask, fill)

    # NDWI — Normalised Difference Water Index
    # (Green - NIR) / (Green + NIR)
    ndwi = safe_ratio(b3 - b8, b3 + b8, mask, fill)

    # NBR — Normalised Burn Ratio
    # (NIR - SWIR2) / (NIR + SWIR2)
    nbr = safe_ratio(b8 - b12, b8 + b12, mask, fill)

    # NDRE — Normalised Difference Red Edge
    # (B8A - B5) / (B8A + B5) — vegetation type discrimination
    ndre = safe_ratio(b8a - b5, b8a + b5, mask, fill)

    # SAVI — Soil Adjusted Vegetation Index
    # 1.5 * (NIR - Red) / (NIR + Red + 0.5) — better for sparse vegetation
    savi = np.where(mask, fill,
                    1.5 * (b8 - b4) / (b8 + b4 + 0.5)
                    ).astype(np.float32)

    # EVI — Enhanced Vegetation Index
    # 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1) — dense canopy
    evi_den = b8 + 6.0 * b4 - 7.5 * b2 + 1.0
    evi = np.where(
        (~mask) & (np.abs(evi_den) > 1e-10),
        2.5 * (b8 - b4) / evi_den,
        fill,
    ).astype(np.float32)
    evi = np.clip(evi, -2.0, 2.0)  # EVI can spike; clip to sensible range

    # NDMI — Normalised Difference Moisture Index
    # (NIR - SWIR1) / (NIR + SWIR1) — canopy water content / wetland detection
    ndmi = safe_ratio(b8 - b11, b8 + b11, mask, fill)

    # BSI — Bare Soil Index
    # ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    bsi_num = (b11 + b4) - (b8 + b2)
    bsi_den = (b11 + b4) + (b8 + b2)
    bsi = safe_ratio(bsi_num, bsi_den, mask, fill)

    return ndvi, ndwi, nbr, ndre, savi, evi, ndmi, bsi


def normalize_band(band, nodata):
    """Min-max normalise to [-1, 1], preserving nodata pixels."""
    nd_mask = nodata_mask(band, nodata)
    valid = band[~nd_mask]

    if valid.size == 0 or valid.max() == valid.min():
        normed = np.zeros_like(band, dtype=np.float32)
    else:
        normed = ((band - valid.min()) / (valid.max() - valid.min()) * 2 - 1).astype(np.float32)

    normed[nd_mask] = nodata if nodata is not None else np.nan
    return normed


def process_image(input_path, output_path):
    with rasterio.open(input_path) as src:
        if src.count != 12:
            print(f"  WARNING: expected 12 bands, got {src.count} — skipping")
            return

        nodata = src.nodata

        # ── Step 1: compute indices ────────────────────────────────────────────
        indices = compute_indices(src, nodata)
        index_names = ["NDVI", "NDWI", "NBR", "NDRE", "SAVI", "EVI", "NDMI", "BSI"]

        # ── Step 2 & 3: normalise kept original bands, drop unwanted bands ─────
        keep_bands = sorted(set(range(1, 13)) - BANDS_TO_REMOVE)  # [2,3,4,5,6,7,8,11,12]

        processed_original = []
        for b in keep_bands:
            band = src.read(b).astype(np.float32)
            processed_original.append(normalize_band(band, nodata))

        # ── Stack: normalised originals + indices ──────────────────────────────
        all_bands = processed_original + list(indices)
        stacked = np.stack(all_bands, axis=0)

        # ── Write output ───────────────────────────────────────────────────────
        out_nodata = nodata if nodata is not None else np.nan
        profile = src.profile.copy()
        profile.update(
            count=stacked.shape[0],
            dtype="float32",
            nodata=out_nodata,
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stacked)


def main():
    input_dir  = Path(cfg.INDICIES_INPUT)
    output_dir = Path(cfg.INDICIES_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    if not images:
        print("No .tif / .tiff files found in input directory.")
        return

    for img_path in sorted(images):
        out_path = output_dir / img_path.name
        print(f"Processing {img_path.name} ...", end=" ", flush=True)
        try:
            process_image(img_path, out_path)
            print("done")
        except Exception as e:
            print(f"FAILED — {e}")

    print(f"\nFinished. {len(images)} image(s) processed → {output_dir}")


if __name__ == "__main__":
    main()
