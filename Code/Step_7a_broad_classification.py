"""
Step 7a — Broad supervised Random Forest classification.

Trains a Random Forest on the `broad_class` field of the training GeoJSON.
Classifies every valid pixel in the AOI into broad habitat groups.

Review the output map before running Step 7b.

Training data GeoJSON must have the field:
    broad_class  — broad habitat group (string, e.g. "Urban", "Grassland")

Outputs (written to SUPERVISED_OUTPUT_DIR):
    classification_broad.tif     — broad classified map (int16, 1-based codes)
    class_mapping_broad.json     — maps integer codes → broad_class label names
    accuracy_stage1_broad.txt    — confusion matrix + metrics

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
    X = data[:, valid_mask].T   # (n_valid_pixels, n_features)

    return X, valid_mask, height, width, profile, transform, crs


def rasterize_labels(gdf, field, le, height, width, transform):
    """
    Rasterize a GeoDataFrame string field to a 1-based int16 raster.
    Smaller polygons painted last so they win overlaps.
    Returns raster where nodata = -1.
    """
    gdf = gdf.copy()
    gdf["_code"] = le.transform(gdf[field]).astype(int) + 1

    areas = gdf.geometry.area.values
    gdf = gdf.iloc[np.argsort(areas)[::-1]]  # largest first, smallest wins

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

    # ── Load training data ─────────────────────────────────────────────────────
    print("\nLoading training data...")
    gdf = gpd.read_file(cfg.SUPERVISED_TRAINING_DATA)

    if "broad_class" not in gdf.columns:
        raise ValueError("Training GeoJSON must have a 'broad_class' field.")

    if gdf.crs is None or gdf.crs != crs:
        print(f"  Reprojecting training data to {crs}...")
        gdf = gdf.to_crs(crs)

    gdf = gdf.dropna(subset=["broad_class"])
    print(f"  {len(gdf)} training polygons | "
          f"{gdf['broad_class'].nunique()} broad classes: {sorted(gdf['broad_class'].unique())}")

    # ── Encode labels ──────────────────────────────────────────────────────────
    le = LabelEncoder().fit(gdf["broad_class"])
    class_names = list(le.classes_)

    mapping = {str(i + 1): name for i, name in enumerate(class_names)}
    mapping_path = output_dir / "class_mapping_broad.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"  Class mapping saved → {mapping_path.name}")

    # ── Rasterize ──────────────────────────────────────────────────────────────
    print("\nRasterizing training polygons...")
    label_raster = rasterize_labels(gdf, "broad_class", le, height, width, transform)

    # ── Build training set ─────────────────────────────────────────────────────
    flat_labels = label_raster.ravel()
    X_full = np.full((flat_valid.size, X.shape[1]), np.nan, dtype="float32")
    X_full[flat_valid] = X

    train_mask = (flat_labels != -1) & flat_valid
    X_labelled = X_full[train_mask]
    y_labelled = flat_labels[train_mask]

    print(f"\n── Training broad Random Forest ──────────────────────────────────────")
    print(f"  Total labelled pixels: {len(y_labelled):,}")
    for code, name in mapping.items():
        count = np.sum(y_labelled == int(code))
        print(f"    {code}: {name:<30} {count:>8,} px")

    # ── 80/20 stratified split ─────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_labelled, y_labelled,
        test_size=1 - cfg.TRAIN_TEST_SPLIT,
        random_state=42,
        stratify=y_labelled,
    )
    print(f"\n  Train: {len(y_tr):,} px  |  Test: {len(y_te):,} px")

    # ── Train ──────────────────────────────────────────────────────────────────
    print("  Fitting Random Forest (200 trees)...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred_te = clf.predict(X_te)
    oa        = accuracy_score(y_te, y_pred_te)
    report    = classification_report(y_te, y_pred_te,
                                      target_names=class_names, zero_division=0)
    cm        = confusion_matrix(y_te, y_pred_te)

    print(f"  Overall accuracy: {oa*100:.2f}%")

    report_str = (
        f"{'='*60}\n"
        f"Stage 1 — Broad Classification\n"
        f"{'='*60}\n"
        f"Train pixels : {len(y_tr):,}\n"
        f"Test pixels  : {len(y_te):,}\n"
        f"Overall Accuracy: {oa:.4f} ({oa*100:.2f}%)\n\n"
        f"Per-class report:\n{report}\n"
        f"Confusion matrix (rows=actual, cols=predicted):\n"
        f"Classes: {class_names}\n{cm}\n"
    )
    save_report(report_str, output_dir / "accuracy_stage1_broad.txt")

    # ── Predict full AOI ───────────────────────────────────────────────────────
    print("\nClassifying full AOI...")
    broad_pred_flat = np.full(flat_valid.size, -1, dtype=np.int16)
    broad_pred_flat[flat_valid] = clf.predict(X).astype(np.int16)
    broad_map = broad_pred_flat.reshape(height, width)

    # ── Export ─────────────────────────────────────────────────────────────────
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="int16", nodata=-1)

    out_path = Path(cfg.BROAD_CLASSIFICATION_OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(broad_map[np.newaxis, ...])

    print(f"\n{'='*60}")
    print(f"Broad classification complete.")
    print(f"  Map            → {out_path}")
    print(f"  Class mapping  → {mapping_path}")
    print(f"\nPixel counts per broad class:")
    unique, counts = np.unique(broad_map[broad_map != -1], return_counts=True)
    for code, count in zip(unique, counts):
        print(f"  {code}: {mapping.get(str(code), '?'):<30} {count:>10,} px")

    print("\nReview the broad classification map, then run Step_7b.")


if __name__ == "__main__":
    main()
