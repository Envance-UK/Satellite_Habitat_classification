"""
Step 8 — Clip classification outputs to catchments and river buffer.

Takes all classification outputs (Steps 6, 7a, 7b) and clips each one to:
    A) Each individual catchment polygon  →  one GeoTIFF per catchment
    B) Each catchment intersected with the 500m river buffer  →  one GeoTIFF per catchment

Output folder structure:
    clipped_outputs/
        <catchment_name>/
            catchment/
                unsupervised/   ← Step 6 k-means outputs
                broad/          ← Step 7a broad classification
                final/          ← Step 7b final classification
            river_buffer/
                unsupervised/
                broad/
                final/

Files are named using the `name` field from the catchments GeoJSON.

Reads paths from Project/Chelmsford_phase_2_config.py.
"""

import importlib.util
import re
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from shapely.ops import unary_union

# ── Load config ────────────────────────────────────────────────────────────────
_config_path = Path(__file__).parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_name(text):
    """Convert a catchment name to a safe filename string."""
    return re.sub(r"[^\w]+", "_", str(text)).strip("_").lower()


def clip_and_save(src_path, geom, out_path):
    """
    Clip a raster to a single Shapely geometry and save.
    Returns True on success, False if geometry doesn't overlap the raster.
    """
    with rasterio.open(src_path) as src:
        try:
            clipped, transform = rio_mask(src, [mapping(geom)], crop=True)
        except ValueError:
            return False  # geometry outside raster extent

        # Skip if result is entirely nodata
        nodata = src.nodata
        if nodata is not None and not np.isnan(float(nodata)):
            if np.all(clipped == nodata):
                return False
        elif np.all(np.isnan(clipped.astype(float))):
            return False

        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)
    return True


def collect_sources():
    """
    Collect all classification GeoTIFFs grouped by type.
    Returns dict: { "unsupervised": [...], "broad": [...], "final": [...] }
    """
    sources = {"unsupervised": [], "broad": [], "final": []}

    # Step 6 — unsupervised k-means (classification_k??.tif)
    unsup_dir = Path(cfg.CLASSIFICATION_OUTPUT_DIR)
    if unsup_dir.exists():
        sources["unsupervised"] = sorted(
            unsup_dir.glob("classification_k*.tif")
        )

    # Step 7a — broad classification
    broad_path = Path(cfg.BROAD_CLASSIFICATION_OUTPUT)
    if broad_path.exists():
        sources["broad"] = [broad_path]

    # Step 7b — final classification
    final_path = Path(cfg.SUPERVISED_OUTPUT_DIR) / "classification_final.tif"
    if final_path.exists():
        sources["final"] = [final_path]

    return sources


def stem_suffix(src_path, classification_type):
    """
    Build the filename suffix from the source file.
    e.g.  classification_k10.tif  →  k10
          classification_broad.tif → broad
          classification_final.tif → final
    """
    stem = src_path.stem  # e.g. "classification_k10"
    suffix = stem.replace("classification_", "")
    return suffix


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    base_dir = Path(cfg.CLIPPED_OUTPUT_DIR)

    # ── Load catchments ────────────────────────────────────────────────────────
    print("Loading catchments...")
    catchments = gpd.read_file(cfg.CATCHMENTS)

    if "name" not in catchments.columns:
        raise ValueError("Catchments GeoJSON must have a 'name' field.")

    print(f"  {len(catchments)} catchments: {list(catchments['name'])}")

    # ── Load river buffer ──────────────────────────────────────────────────────
    print("Loading river buffer...")
    buffer_gdf  = gpd.read_file(cfg.RIVER_BUFFER)
    buffer_geom = unary_union(buffer_gdf.geometry)   # single dissolved polygon

    # ── Collect source files ───────────────────────────────────────────────────
    sources = collect_sources()
    total = sum(len(v) for v in sources.values())
    if total == 0:
        print("No classification files found. Run Steps 6, 7a, and 7b first.")
        return

    print(f"\nClassification files found:")
    for group, paths in sources.items():
        print(f"  {group:<15} {len(paths)} file(s): "
              f"{[p.name for p in paths]}")

    # ── Process each catchment ────────────────────────────────────────────────
    print(f"\nClipping to {len(catchments)} catchments × "
          f"{total} files × 2 clip types...\n")

    for _, row in catchments.iterrows():
        catchment_name = safe_name(row["name"])
        catchment_geom = row.geometry

        # Intersect catchment with river buffer
        buffer_intersection = catchment_geom.intersection(buffer_geom)
        has_buffer = not buffer_intersection.is_empty

        print(f"── {row['name']} ─────────────────────────────────────")

        catchment_dir = base_dir / catchment_name

        for group, src_paths in sources.items():
            for src_path in src_paths:
                suffix = stem_suffix(src_path, group)

                # A) Clip to catchment
                out_catchment = (
                    catchment_dir / "catchment" / group
                    / f"{catchment_name}_{suffix}.tif"
                )
                ok = clip_and_save(src_path, catchment_geom, out_catchment)
                status = "saved" if ok else "no overlap"
                print(f"  catchment | {group:<15} {suffix:<10} → {status}")

                # B) Clip to catchment ∩ river buffer
                if has_buffer:
                    out_buffer = (
                        catchment_dir / "river_buffer" / group
                        / f"{catchment_name}_{suffix}.tif"
                    )
                    ok = clip_and_save(src_path, buffer_intersection, out_buffer)
                    status = "saved" if ok else "no overlap"
                    print(f"  buffer    | {group:<15} {suffix:<10} → {status}")
                else:
                    print(f"  buffer    | {group:<15} {suffix:<10} → "
                          "no buffer overlap for this catchment")

    print(f"\n{'='*60}")
    print(f"Done. All outputs saved to: {base_dir}")
    print(f"\nFolder structure:")
    for _, row in catchments.iterrows():
        name = safe_name(row["name"])
        print(f"  {name}/")
        print(f"    ├── catchment/")
        print(f"    │     ├── unsupervised/")
        print(f"    │     ├── broad/")
        print(f"    │     └── final/")
        print(f"    └── river_buffer/")
        print(f"          ├── unsupervised/")
        print(f"          ├── broad/")
        print(f"          └── final/")


if __name__ == "__main__":
    main()
