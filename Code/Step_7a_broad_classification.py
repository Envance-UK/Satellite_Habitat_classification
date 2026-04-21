"""
Step 7a — Vegetation/Non-Vegetation split, then broad classification.

Three-stage process:
    Stage 1 — Veg / Non-Veg classification
        Derives veg/non-veg labels from `broad_class` using the lists below.
        Trains a Random Forest and classifies every valid pixel.
        Outputs: classification_veg_nonveg.tif

    Stage 2a — Broad classification within Vegetation pixels
        Trains a separate Random Forest on vegetation broad classes only.

    Stage 2b — Broad classification within Non-Vegetation pixels
        Trains a separate Random Forest on non-vegetation broad classes only.

        Both broad classifiers are combined into: classification_broad.tif

Training data GeoJSON must have the field:
    broad_class  — broad habitat group (string, must match the lists below)

Outputs (written to SUPERVISED_OUTPUT_DIR):
    classification_veg_nonveg.tif   — veg/non-veg map (1=Vegetation, 2=Non_Vegetation)
    classification_broad.tif        — broad classified map (int16, 1-based codes)
    class_mapping_veg_nonveg.json   — code → label for veg/non-veg map
    class_mapping_broad.json        — code → label for broad map

Reads paths from Project/Chelmsford_phase_2_config.py.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── Load config ────────────────────────────────────────────────────────────────
_config_path = Path(__file__).parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

# Class lists are defined in Chelmsford_phase_2_config.py
VEGETATION_CLASSES     = cfg.VEGETATION_CLASSES
NON_VEGETATION_CLASSES = cfg.NON_VEGETATION_CLASSES


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_features(stats_tif):
    """Load statistics GeoTIFF → feature matrix and spatial metadata."""
    with rasterio.open(stats_tif) as src:
        data      = src.read().astype("float32")
        profile   = src.profile.copy()
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata

    _, height, width = data.shape

    if nodata is not None and not np.isnan(float(nodata)):
        invalid = np.any((data == nodata) | np.isnan(data), axis=0)
    else:
        invalid = np.any(np.isnan(data), axis=0)

    valid_mask = ~invalid
    X = data[:, valid_mask].T   # (n_valid_pixels, n_features)

    return X, valid_mask, height, width, profile, transform, crs


def rasterize_field(gdf, code_col, height, width, transform):
    """
    Rasterize a pre-computed integer code column.
    Smaller polygons painted last so they win overlaps.
    Returns int16 raster, nodata = -1.
    """
    gdf = gdf.copy().iloc[np.argsort(gdf.geometry.area.values)[::-1]]
    shapes = (
        (geom, int(code))
        for geom, code in zip(gdf.geometry, gdf[code_col])
        if geom is not None and not geom.is_empty
    )
    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,
        dtype=np.int16,
    )


def train_rf(X, y):
    """Train a Random Forest on all labelled pixels."""
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf


def save_tif(arr, profile, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr[np.newaxis, ...])


def print_pixel_counts(flat_map, mapping, label):
    print(f"\n  Pixel counts — {label}:")
    unique, counts = np.unique(flat_map[flat_map != -1], return_counts=True)
    for code, count in zip(unique, counts):
        print(f"    {code}: {mapping.get(str(code), '?'):<35} {count:>10,} px")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path(cfg.SUPERVISED_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate class lists ───────────────────────────────────────────────────
    overlap = set(VEGETATION_CLASSES) & set(NON_VEGETATION_CLASSES)
    if overlap:
        raise ValueError(f"Classes appear in both lists: {overlap}")

    # ── Load features ──────────────────────────────────────────────────────────
    print("Loading statistics GeoTIFF...")
    X, valid_mask, height, width, profile, transform, crs = load_features(
        Path(cfg.SUPERVISED_INPUT)
    )
    print(f"  Valid pixels: {X.shape[0]:,}  |  Features: {X.shape[1]}")

    flat_valid = valid_mask.ravel()
    X_full = np.full((flat_valid.size, X.shape[1]), np.nan, dtype="float32")
    X_full[flat_valid] = X

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="int16", nodata=-1)

    # ── Load training data ─────────────────────────────────────────────────────
    print("\nLoading training data...")
    gdf = gpd.read_file(cfg.SUPERVISED_TRAINING_DATA)

    if "broad_class" not in gdf.columns:
        raise ValueError("Training GeoJSON must have a 'broad_class' field.")

    if gdf.crs is None or gdf.crs != crs:
        print(f"  Reprojecting training data to {crs}...")
        gdf = gdf.to_crs(crs)

    gdf = gdf.dropna(subset=["broad_class"])

    # Warn if any broad_class values aren't in either list
    all_classes  = set(VEGETATION_CLASSES) | set(NON_VEGETATION_CLASSES)
    found_classes = set(gdf["broad_class"].unique())
    unknown = found_classes - all_classes
    if unknown:
        print(f"  WARNING: these broad_class values are not in VEG or NON_VEG lists "
              f"and will be ignored: {unknown}")

    gdf = gdf[gdf["broad_class"].isin(all_classes)].copy()
    print(f"  {len(gdf)} training polygons | "
          f"classes: {sorted(gdf['broad_class'].unique())}")

    # ── Assign veg/non-veg codes ───────────────────────────────────────────────
    # 1 = Vegetation, 2 = Non_Vegetation
    VEG_CODE     = 1
    NON_VEG_CODE = 2
    gdf["_veg_code"] = gdf["broad_class"].apply(
        lambda c: VEG_CODE if c in VEGETATION_CLASSES else NON_VEG_CODE
    )

    veg_nonveg_mapping = {"1": "Vegetation", "2": "Non_Vegetation"}
    (output_dir / "class_mapping_veg_nonveg.json").write_text(
        json.dumps(veg_nonveg_mapping, indent=2), encoding="utf-8"
    )

    # ── Broad class encoder (global, for final output codes) ──────────────────
    le_broad = LabelEncoder().fit(sorted(gdf["broad_class"].unique()))
    broad_mapping = {str(i + 1): name for i, name in enumerate(le_broad.classes_)}
    (output_dir / "class_mapping_broad.json").write_text(
        json.dumps(broad_mapping, indent=2), encoding="utf-8"
    )
    print(f"\n  Broad class mapping:")
    for code, name in broad_mapping.items():
        grp = "VEG" if name in VEGETATION_CLASSES else "NON-VEG"
        print(f"    {code}: {name:<30} [{grp}]")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Veg / Non-Veg classification
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Stage 1: Vegetation / Non-Vegetation ──────────────────────────────")

    veg_raster  = rasterize_field(gdf, "_veg_code", height, width, transform)
    flat_veg    = veg_raster.ravel()
    train_mask  = (flat_veg != -1) & flat_valid

    X_tr = X_full[train_mask]
    y_tr = flat_veg[train_mask]

    print(f"  Labelled pixels:  {len(y_tr):,}")
    print(f"    Vegetation:     {np.sum(y_tr == VEG_CODE):>10,} px")
    print(f"    Non-Vegetation: {np.sum(y_tr == NON_VEG_CODE):>10,} px")
    print("  Fitting Veg/Non-Veg Random Forest...")

    clf_veg = train_rf(X_tr, y_tr)

    veg_pred_flat = np.full(flat_valid.size, -1, dtype=np.int16)
    veg_pred_flat[flat_valid] = clf_veg.predict(X).astype(np.int16)
    veg_map = veg_pred_flat.reshape(height, width)

    veg_tif = output_dir / "classification_veg_nonveg.tif"
    save_tif(veg_map, out_profile, veg_tif)
    print(f"  Saved → {veg_tif.name}")
    print_pixel_counts(veg_pred_flat, veg_nonveg_mapping, "Veg/Non-Veg")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Broad classification (one RF per veg group)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Stage 2: Broad classification ─────────────────────────────────────")

    # Rasterize broad_class using global encoder
    gdf["_broad_code"] = le_broad.transform(gdf["broad_class"]).astype(int) + 1
    broad_raster = rasterize_field(gdf, "_broad_code", height, width, transform)
    flat_broad   = broad_raster.ravel()

    broad_pred_flat = np.full(flat_valid.size, -1, dtype=np.int16)

    for group_label, veg_code, class_list in [
        ("Vegetation",     VEG_CODE,     VEGETATION_CLASSES),
        ("Non-Vegetation", NON_VEG_CODE, NON_VEGETATION_CLASSES),
    ]:
        print(f"\n  [{group_label}]")

        # Pixels the veg classifier assigned to this group
        group_pred_mask = (veg_pred_flat == veg_code)

        # Training pixels: labelled with a broad class in this group AND valid
        group_train_classes = [
            c for c in class_list if c in found_classes
        ]
        if not group_train_classes:
            print(f"  WARNING: no training polygons for {group_label} — skipping.")
            continue

        group_codes = set(
            le_broad.transform(group_train_classes).astype(int) + 1
        )
        train_mask_broad = (
            np.isin(flat_broad, list(group_codes)) & flat_valid
        )

        X_tr = X_full[train_mask_broad]
        y_tr = flat_broad[train_mask_broad]

        print(f"  Labelled pixels: {len(y_tr):,}")
        for code in sorted(group_codes):
            name  = broad_mapping.get(str(code), "?")
            count = np.sum(y_tr == code)
            print(f"    {code}: {name:<30} {count:>8,} px")

        if len(np.unique(y_tr)) < 1:
            print("  WARNING: no valid training pixels — skipping.")
            continue

        print(f"  Fitting {group_label} broad Random Forest...")
        clf_broad = train_rf(X_tr, y_tr)

        # Predict only on pixels assigned to this veg group
        X_group = X_full[group_pred_mask]
        broad_pred_flat[group_pred_mask] = clf_broad.predict(X_group).astype(np.int16)

    broad_map = broad_pred_flat.reshape(height, width)

    broad_tif = Path(cfg.BROAD_CLASSIFICATION_OUTPUT)
    save_tif(broad_map, out_profile, broad_tif)
    print(f"\n  Saved → {broad_tif.name}")
    print_pixel_counts(broad_pred_flat, broad_mapping, "Broad classes")

    print(f"\n{'='*60}")
    print("Step 7a complete.")
    print(f"  Veg/Non-Veg map → {veg_tif.name}")
    print(f"  Broad map       → {broad_tif.name}")
    print(f"  Mappings        → class_mapping_veg_nonveg.json, class_mapping_broad.json")
    print("\nReview both maps, then run Step_7b.")


if __name__ == "__main__":
    main()
