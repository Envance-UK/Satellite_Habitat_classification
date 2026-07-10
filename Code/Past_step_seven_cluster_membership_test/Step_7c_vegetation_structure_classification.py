"""
Past_step_seven_cluster_membership_test — Step 7c — Vegetation structure
classification (id-based), with k-means cluster membership added as an extra
Random Forest feature.

Identical to Past_step_seven_v2/Step_7c_vegetation_structure_classification.py,
except the feature matrix also includes the k-means cluster id from Step 6
(CLUSTER_MEMBERSHIP_TIF in config) as one extra column alongside the 102
statistics bands.

Reads the main habitat classification map produced by this folder's Step 7b,
then for each main_id that has Vegetation_Structure training data, trains a
separate Random Forest on structure_id and applies it only to pixels assigned
to that main_id.

Main classes with no Vegetation_Structure training data (e.g. Open Water, Urban)
are left as nodata in the output — there is no structure to classify there.

Output pixel values ARE your structure_id values directly (not re-encoded).

Must be run after Past_step_seven_cluster_membership_test/Step_7b_main_classification.py.

Training data GeoJSON must have the fields:
    main_id, Vegetation_Structure, structure_id

Outputs (written to CLUSTER_TEST_OUTPUT_DIR):
    classification_vegetation_structure_cluster.tif — structure map (int16, values = your structure_id)
    class_mapping_vegetation_structure.json     — structure_id -> Vegetation_Structure
    accuracy_stage_structure_<main_habitat>.txt — one accuracy report per main class

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

# ── Load config ────────────────────────────────────────────────────────────────
_config_path = Path(__file__).parent.parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_features(stats_tif, cluster_tif):
    """
    Load the statistics GeoTIFF plus a k-means cluster-membership GeoTIFF
    (from Step 6) and return a combined feature matrix — statistics bands
    with the cluster id appended as one extra feature — plus spatial metadata.
    """
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

    with rasterio.open(cluster_tif) as src:
        cluster = src.read(1).astype("float32")

    if cluster.shape != (height, width):
        raise ValueError(
            f"CLUSTER_MEMBERSHIP_TIF shape {cluster.shape} does not match the "
            f"statistics tif shape {(height, width)} — it must be a k-means "
            f"output derived from the same statistics GeoTIFF (Step 5/6)."
        )

    # Step 6's K-means output uses 0 as nodata for invalid pixels — any pixel
    # valid in the stats tif but nodata in the cluster tif is a mismatch.
    mismatched = valid_mask & (cluster == 0)
    if mismatched.any():
        print(f"  WARNING: {mismatched.sum():,} pixels are valid in the statistics "
              f"tif but have no cluster assigned — excluding them from training/prediction.")
        valid_mask = valid_mask & (cluster != 0)

    X_stats   = data[:, valid_mask].T
    X_cluster = cluster[valid_mask][:, np.newaxis]
    X = np.hstack([X_stats, X_cluster])

    return X, valid_mask, height, width, profile, transform, crs


def rasterize_field(gdf, code_col, height, width, transform):
    """Rasterize a pre-computed integer code column. Returns int16 raster, nodata = -1."""
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


def write_class_labels(tif_path, mapping):
    """
    Attach id -> label pairs to a classified GeoTIFF as a GDAL Raster
    Attribute Table (written to a `<tif>.aux.xml` sidecar file), so GIS
    software like QGIS/ArcGIS can show a labeled attribute table / legend.

    Falls back to embedding the mapping as a plain band metadata tag if
    osgeo (GDAL Python bindings) isn't installed.
    """
    try:
        from osgeo import gdal
    except ImportError:
        with rasterio.open(tif_path, "r+") as dst:
            dst.update_tags(1, CLASS_MAPPING=json.dumps(
                {str(k): v for k, v in mapping.items()}
            ))
        print(f"  osgeo not available — labels embedded as a metadata tag on "
              f"{tif_path.name} (no attribute table / auto legend in GIS software).")
        return

    ds   = gdal.Open(str(tif_path), gdal.GA_Update)
    band = ds.GetRasterBand(1)

    rat = gdal.RasterAttributeTable()
    rat.CreateColumn("Value", gdal.GFT_Integer, gdal.GFU_MinMax)
    rat.CreateColumn("Label", gdal.GFT_String, gdal.GFU_Name)

    rows = sorted(mapping.items())
    rat.SetRowCount(len(rows))
    for i, (code, label) in enumerate(rows):
        rat.SetValueAsInt(i, 0, int(code))
        rat.SetValueAsString(i, 1, str(label))

    band.SetDefaultRAT(rat)
    band.FlushCache()
    ds.FlushCache()
    ds = None
    print(f"  Raster Attribute Table written → {tif_path.name}.aux.xml")


def build_id_name_mapping(gdf, id_col, name_col):
    """Build {id: name} from unique pairs, checking the mapping is 1-to-1."""
    pairs = gdf[[id_col, name_col]].drop_duplicates()
    mapping = {}
    for _, row in pairs.iterrows():
        code = int(row[id_col])
        name = row[name_col]
        if code in mapping and mapping[code] != name:
            raise ValueError(
                f"{id_col}={code} maps to both '{mapping[code]}' and '{name}' — "
                f"ids must be 1-to-1 with {name_col}."
            )
        mapping[code] = name
    return mapping


def save_report(report_str, path):
    path.write_text(report_str, encoding="utf-8")
    print(f"  Accuracy report → {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path(cfg.CLUSTER_TEST_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.CLUSTER_MEMBERSHIP_TIF:
        raise ValueError(
            "CLUSTER_MEMBERSHIP_TIF is empty in config — set it to one of the "
            "classification_k##.tif files produced by Step 6."
        )

    # ── Load features ──────────────────────────────────────────────────────────
    print("Loading statistics GeoTIFF + cluster membership...")
    X, valid_mask, height, width, profile, transform, crs = load_features(
        Path(cfg.SUPERVISED_INPUT), Path(cfg.CLUSTER_MEMBERSHIP_TIF)
    )
    print(f"  Valid pixels: {X.shape[0]:,}  |  Features: {X.shape[1]} "
          f"(includes cluster membership)")

    flat_valid = valid_mask.ravel()
    X_full = np.full((flat_valid.size, X.shape[1]), np.nan, dtype="float32")
    X_full[flat_valid] = X

    # ── Load main classification map from this folder's Step 7b ───────────────
    main_tif = output_dir / "classification_main_cluster.tif"
    if not main_tif.exists():
        raise FileNotFoundError(
            f"Main classification map not found: {main_tif}\n"
            "Run Past_step_seven_cluster_membership_test/Step_7b_main_classification.py first."
        )
    print(f"\nLoading main classification map: {main_tif.name}")
    with rasterio.open(main_tif) as src:
        main_flat = src.read(1).ravel()

    main_mapping_path = output_dir / "class_mapping_main.json"
    if not main_mapping_path.exists():
        raise FileNotFoundError(
            f"Main class mapping not found: {main_mapping_path}\n"
            "Run Past_step_seven_cluster_membership_test/Step_7b_main_classification.py first."
        )
    with open(main_mapping_path) as f:
        main_mapping = {int(k): v for k, v in json.load(f).items()}
    print(f"  Main classes: {main_mapping}")

    # ── Load training data ───────────────────────────────────────────────────────
    print("\nLoading training data...")
    gdf = gpd.read_file(cfg.V2_TRAINING_DATA)

    required = {"main_id", "Vegetation_Structure", "structure_id"}
    if not required.issubset(gdf.columns):
        raise ValueError(f"Training GeoJSON must have fields: {required}")

    if gdf.crs is None or gdf.crs != crs:
        print(f"  Reprojecting training data to {crs}...")
        gdf = gdf.to_crs(crs)

    # Only rows with a structure value can train Step 7c — main classes with no
    # Vegetation_Structure entries (e.g. Open Water) are naturally excluded here.
    gdf = gdf.dropna(subset=["main_id", "Vegetation_Structure", "structure_id"]).copy()
    gdf["main_id"]      = gdf["main_id"].astype(int)
    gdf["structure_id"] = gdf["structure_id"].astype(int)

    if (gdf["structure_id"] < 0).any():
        raise ValueError("structure_id must be >= 0 (-1 is reserved for nodata).")

    structure_mapping = build_id_name_mapping(gdf, "structure_id", "Vegetation_Structure")
    mapping_path = output_dir / "class_mapping_vegetation_structure.json"
    mapping_path.write_text(
        json.dumps({str(k): v for k, v in sorted(structure_mapping.items())}, indent=2),
        encoding="utf-8",
    )
    print(f"  {len(structure_mapping)} vegetation structure classes | "
          f"mapping → {mapping_path.name}")

    # ── Final output initialised to nodata ─────────────────────────────────────
    final_flat = np.full(flat_valid.size, -1, dtype=np.int16)

    # ══════════════════════════════════════════════════════════════════════════
    # For each main_id with structure training data → train RF on structure_id
    # ══════════════════════════════════════════════════════════════════════════
    for main_id, main_label in sorted(main_mapping.items()):
        print(f"\n── Main class {main_id}: '{main_label}' ──────────────────────────")

        # Pixels assigned to this main class by Step 7b
        pred_mask_flat = (main_flat == main_id)
        n_pred = pred_mask_flat.sum()
        print(f"  Pixels to classify: {n_pred:,}")

        if n_pred == 0:
            print("  No pixels predicted for this main class — skipping.")
            continue

        sub_gdf = gdf[gdf["main_id"] == main_id].copy()

        if sub_gdf.empty:
            print(f"  No Vegetation_Structure training data for '{main_label}' — "
                  "leaving these pixels as nodata (no structure to classify).")
            continue

        sub_ids = sorted(sub_gdf["structure_id"].unique())
        print(f"  Sub-classes ({len(sub_ids)}): "
              f"{[(i, structure_mapping[i]) for i in sub_ids]}")

        if len(sub_ids) < 2:
            print(f"  WARNING: only 1 structure class for '{main_label}' — "
                  "assigning directly without classifier.")
            final_flat[pred_mask_flat] = sub_ids[0]
            continue

        # Rasterize structure_id for this main group only
        sub_raster  = rasterize_field(sub_gdf, "structure_id", height, width, transform)
        flat_labels = sub_raster.ravel()

        train_mask = (flat_labels != -1) & flat_valid
        X_labelled = X_full[train_mask]
        y_labelled = flat_labels[train_mask]

        if len(y_labelled) == 0:
            print("  WARNING: no valid labelled pixels found — skipping.")
            continue

        print(f"  Labelled pixels available: {len(y_labelled):,}")
        for code in sub_ids:
            count = np.sum(y_labelled == code)
            print(f"    {code}: {structure_mapping[code]:<35} {count:>8,} px")

        # 80/20 stratified split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_labelled, y_labelled,
            test_size=1 - cfg.TRAIN_TEST_SPLIT,
            random_state=42,
            stratify=y_labelled,
        )
        print(f"  Train: {len(y_tr):,}  |  Test: {len(y_te):,}")

        print("  Fitting Random Forest (200 trees)...")
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        # Evaluate on test set
        y_pred_te = clf.predict(X_te)
        oa        = accuracy_score(y_te, y_pred_te)
        labels_present = sorted(set(y_tr) | set(y_te))
        report = classification_report(
            y_te, y_pred_te,
            labels=labels_present,
            target_names=[structure_mapping[i] for i in labels_present],
            zero_division=0,
        )
        cm = confusion_matrix(y_te, y_pred_te, labels=labels_present)

        print(f"  Overall accuracy: {oa*100:.2f}%")

        report_str = (
            f"{'='*60}\n"
            f"Step 7c — Vegetation structure: {main_label}\n"
            f"{'='*60}\n"
            f"Train pixels : {len(y_tr):,}\n"
            f"Test pixels  : {len(y_te):,}\n"
            f"Overall Accuracy: {oa:.4f} ({oa*100:.2f}%)\n\n"
            f"Per-class report:\n{report}\n"
            f"Confusion matrix (rows=actual, cols=predicted):\n"
            f"Classes: {[structure_mapping[i] for i in labels_present]}\n{cm}\n"
        )
        report_name = f"accuracy_stage_structure_{main_label.replace(' ', '_').lower()}.txt"
        save_report(report_str, output_dir / report_name)

        # Predict on all pixels assigned to this main class
        X_pred = X_full[pred_mask_flat]
        final_flat[pred_mask_flat] = clf.predict(X_pred).astype(np.int16)

    # ── Export final classification ────────────────────────────────────────────
    final_map = final_flat.reshape(height, width)

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="int16", nodata=-1)

    out_path = output_dir / "classification_vegetation_structure_cluster.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(final_map[np.newaxis, ...])
    write_class_labels(out_path, structure_mapping)

    print(f"\n{'='*60}")
    print("Step 7c complete.")
    print(f"  Vegetation structure map → {out_path.name}")
    print(f"  Class mapping            → {mapping_path.name}")

    print("\nPixel counts per vegetation structure class:")
    unique, counts = np.unique(final_map[final_map != -1], return_counts=True)
    for code, count in zip(unique, counts):
        label = structure_mapping.get(int(code), f"code_{code}")
        print(f"  {code:3d}  {label:<35} {count:>10,} px")


if __name__ == "__main__":
    main()
