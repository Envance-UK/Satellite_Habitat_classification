"""
Step 7b — Refined (main habitat) supervised Random Forest classification.

Reads the broad classification map produced by Step 7a, then for each broad
class trains a separate Random Forest on the `main_habitat` field and applies
it only to pixels that were assigned to that broad group.

Must be run after Step_7a_broad_classification.py.

Training data GeoJSON must have the fields:
    broad_class   — matches the broad groups from Step 7a
    main_habitat  — fine habitat type (e.g. "Improved Grassland", "Rush Pasture")

Outputs (written to SUPERVISED_OUTPUT_DIR):
    classification_final.tif            — final classified map (int16, 1-based codes)
    class_mapping_main.json             — maps integer codes → main_habitat label names
    accuracy_stage2_<broad_class>.txt   — one accuracy report per broad class

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── Load config ────────────────────────────────────────────────────────────────
_config_path = Path(__file__).parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)


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
    X = data[:, valid_mask].T

    return X, valid_mask, height, width, profile, transform, crs


def load_broad_map(broad_tif):
    """Load the broad classification GeoTIFF from Step 7a."""
    with rasterio.open(broad_tif) as src:
        broad_map = src.read(1)
    return broad_map


def rasterize_labels(gdf, field, le, height, width, transform):
    """Rasterize a string field to a 1-based int16 raster. Nodata = -1."""
    gdf = gdf.copy()
    gdf["_code"] = le.transform(gdf[field]).astype(int) + 1

    areas = gdf.geometry.area.values
    gdf = gdf.iloc[np.argsort(areas)[::-1]]

    shapes = (
        (geom, int(code))
        for geom, code in zip(gdf.geometry, gdf["_code"])
        if geom is not None and not geom.is_empty
    )
    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,
        dtype=np.int16,
    )


def save_report(report_str, path):
    path.write_text(report_str, encoding="utf-8")
    print(f"  Accuracy report → {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path(cfg.SUPERVISED_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load features ──────────────────────────────────────────────────────────
    print("Loading statistics GeoTIFF...")
    X, valid_mask, height, width, profile, transform, crs = load_features(
        Path(cfg.SUPERVISED_INPUT)
    )
    print(f"  Valid pixels: {X.shape[0]:,}  |  Features: {X.shape[1]}")

    flat_valid = valid_mask.ravel()

    # Expand X to full pixel space for indexed access
    X_full = np.full((flat_valid.size, X.shape[1]), np.nan, dtype="float32")
    X_full[flat_valid] = X

    # ── Load broad classification map from Step 7a ─────────────────────────────
    broad_tif = Path(cfg.BROAD_CLASSIFICATION_OUTPUT)
    if not broad_tif.exists():
        raise FileNotFoundError(
            f"Broad classification map not found: {broad_tif}\n"
            "Run Step_7a_broad_classification.py first."
        )
    print(f"\nLoading broad classification map: {broad_tif.name}")
    broad_map  = load_broad_map(broad_tif)
    broad_flat = broad_map.ravel()

    # Load broad class mapping to get label names
    broad_mapping_path = output_dir / "class_mapping_broad.json"
    if not broad_mapping_path.exists():
        raise FileNotFoundError(
            f"Broad class mapping not found: {broad_mapping_path}\n"
            "Run Step_7a_broad_classification.py first."
        )
    with open(broad_mapping_path) as f:
        broad_mapping = json.load(f)   # {"1": "Grassland", "2": "Urban", ...}

    print(f"  Broad classes: { {v: k for k, v in broad_mapping.items()} }")

    # ── Load training data ─────────────────────────────────────────────────────
    print("\nLoading training data...")
    gdf = gpd.read_file(cfg.SUPERVISED_TRAINING_DATA)

    required = {"broad_class", "main_habitat"}
    if not required.issubset(gdf.columns):
        raise ValueError(f"Training GeoJSON must have fields: {required}")

    if gdf.crs is None or gdf.crs != crs:
        print(f"  Reprojecting training data to {crs}...")
        gdf = gdf.to_crs(crs)

    gdf = gdf.dropna(subset=["broad_class", "main_habitat"])
    print(f"  {len(gdf)} training polygons")

    # ── Global main_habitat encoder (for consistent output codes) ──────────────
    le_global = LabelEncoder().fit(gdf["main_habitat"])
    global_names = list(le_global.classes_)
    main_mapping = {str(i + 1): name for i, name in enumerate(global_names)}

    mapping_path = output_dir / "class_mapping_main.json"
    mapping_path.write_text(json.dumps(main_mapping, indent=2), encoding="utf-8")
    print(f"  {len(global_names)} main habitat classes | mapping → {mapping_path.name}")

    # ── Final output initialised to nodata ─────────────────────────────────────
    final_flat = np.full(flat_valid.size, -1, dtype=np.int16)

    # ══════════════════════════════════════════════════════════════════════════
    # For each broad class → train refined RF → predict
    # ══════════════════════════════════════════════════════════════════════════
    for broad_code_str, broad_label in broad_mapping.items():
        broad_code = int(broad_code_str)
        print(f"\n── Broad class {broad_code}: '{broad_label}' ──────────────────────────")

        # Pixels assigned to this broad class by Step 7a
        pred_mask_flat = (broad_flat == broad_code)
        n_pred = pred_mask_flat.sum()
        print(f"  Pixels to classify: {n_pred:,}")

        if n_pred == 0:
            print("  No pixels predicted for this broad class — skipping.")
            continue

        # Training polygons for this broad class
        sub_gdf = gdf[gdf["broad_class"] == broad_label].copy()

        if sub_gdf.empty:
            print(f"  WARNING: no training polygons for '{broad_label}' — "
                  "assigning broad code to all pixels in this group.")
            final_flat[pred_mask_flat] = broad_code
            continue

        sub_classes = sorted(sub_gdf["main_habitat"].unique())
        print(f"  Sub-classes ({len(sub_classes)}): {sub_classes}")

        if len(sub_classes) < 2:
            print(f"  WARNING: only 1 sub-class for '{broad_label}' — "
                  "assigning directly without classifier.")
            global_code = le_global.transform([sub_classes[0]])[0] + 1
            final_flat[pred_mask_flat] = global_code
            continue

        # Local encoder for this broad group's sub-classes
        le_sub = LabelEncoder().fit(sub_classes)

        # Rasterize main_habitat for this broad group only
        sub_raster  = rasterize_labels(sub_gdf, "main_habitat", le_sub,
                                       height, width, transform)
        flat_labels = sub_raster.ravel()

        # Training pixels: labelled AND valid features
        train_mask = (flat_labels != -1) & flat_valid
        X_labelled = X_full[train_mask]
        y_labelled = flat_labels[train_mask]

        if len(y_labelled) == 0:
            print("  WARNING: no valid labelled pixels found — skipping.")
            continue

        print(f"  Labelled pixels available: {len(y_labelled):,}")
        for local_code, name in enumerate(le_sub.classes_, start=1):
            count = np.sum(y_labelled == local_code)
            print(f"    {local_code}: {name:<35} {count:>8,} px")

        # 80/20 stratified split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_labelled, y_labelled,
            test_size=1 - cfg.TRAIN_TEST_SPLIT,
            random_state=42,
            stratify=y_labelled,
        )
        print(f"  Train: {len(y_tr):,}  |  Test: {len(y_te):,}")

        # Train
        print("  Fitting Random Forest (200 trees)...")
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        # Evaluate on test set
        y_pred_te = clf.predict(X_te)
        oa        = accuracy_score(y_te, y_pred_te)
        report    = classification_report(y_te, y_pred_te,
                                          target_names=list(le_sub.classes_),
                                          zero_division=0)
        cm        = confusion_matrix(y_te, y_pred_te)

        print(f"  Overall accuracy: {oa*100:.2f}%")

        report_str = (
            f"{'='*60}\n"
            f"Stage 2 — Refined: {broad_label}\n"
            f"{'='*60}\n"
            f"Train pixels : {len(y_tr):,}\n"
            f"Test pixels  : {len(y_te):,}\n"
            f"Overall Accuracy: {oa:.4f} ({oa*100:.2f}%)\n\n"
            f"Per-class report:\n{report}\n"
            f"Confusion matrix (rows=actual, cols=predicted):\n"
            f"Classes: {list(le_sub.classes_)}\n{cm}\n"
        )
        report_name = f"accuracy_stage2_{broad_label.replace(' ', '_').lower()}.txt"
        save_report(report_str, output_dir / report_name)

        # Predict on all pixels assigned to this broad class
        X_pred   = X_full[pred_mask_flat]
        sub_pred = clf.predict(X_pred).astype(np.int16)   # local 1-based codes

        # Map local codes → global main_habitat codes
        global_codes = np.array(
            [le_global.transform([label])[0] + 1 for label in le_sub.classes_],
            dtype=np.int16,
        )
        final_flat[pred_mask_flat] = global_codes[sub_pred - 1]

    # ── Export final classification ────────────────────────────────────────────
    final_map = final_flat.reshape(height, width)

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="int16", nodata=-1)

    out_path = output_dir / "classification_final.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(final_map[np.newaxis, ...])

    print(f"\n{'='*60}")
    print(f"Main classification complete.")
    print(f"  Final map      → {out_path.name}")
    print(f"  Class mapping  → {mapping_path.name}")

    print("\nPixel counts per main habitat class:")
    unique, counts = np.unique(final_map[final_map != -1], return_counts=True)
    for code, count in zip(unique, counts):
        label = main_mapping.get(str(code), f"code_{code}")
        print(f"  {code:3d}  {label:<35} {count:>10,} px")


if __name__ == "__main__":
    main()
