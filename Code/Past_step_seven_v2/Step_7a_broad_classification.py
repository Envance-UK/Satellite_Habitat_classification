"""
Past_step_seven_v2 — Step 7a — Veg/Non-Veg split, then broad classification (id-based).

Three-stage process:
    Stage 1 — Veg / Non-Veg classification
        Groups broad_id values into Vegetation / Non-Vegetation using
        VEGETATION_BROAD_IDS / NON_VEGETATION_BROAD_IDS (set these in config).
        Trains a Random Forest and classifies every valid pixel.
        Outputs: classification_veg_nonveg.tif

    Stage 2a — Broad classification within Vegetation pixels
    Stage 2b — Broad classification within Non-Vegetation pixels
        Trains a separate Random Forest per group on broad_id.
        Output pixel values ARE your broad_id values directly (not re-encoded).

Training data GeoJSON must have the fields:
    broad_habitat_type, broad_id

Outputs (written to V2_OUTPUT_DIR):
    classification_veg_nonveg.tif   — veg/non-veg map (1=Vegetation, 2=Non_Vegetation)
    classification_broad.tif        — broad classified map (int16, values = your broad_id)
    class_mapping_veg_nonveg.json   — code -> label for veg/non-veg map
    class_mapping_broad.json        — broad_id -> broad_habitat_type

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

# ── Load config ────────────────────────────────────────────────────────────────
_config_path = Path(__file__).parent.parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

VEGETATION_BROAD_IDS     = cfg.VEGETATION_BROAD_IDS
NON_VEGETATION_BROAD_IDS = cfg.NON_VEGETATION_BROAD_IDS


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


def print_pixel_counts(flat_map, mapping, label):
    print(f"\n  Pixel counts — {label}:")
    unique, counts = np.unique(flat_map[flat_map != -1], return_counts=True)
    for code, count in zip(unique, counts):
        print(f"    {code}: {mapping.get(int(code), '?'):<35} {count:>10,} px")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path(cfg.V2_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate id lists ───────────────────────────────────────────────────────
    if not VEGETATION_BROAD_IDS or not NON_VEGETATION_BROAD_IDS:
        raise ValueError(
            "VEGETATION_BROAD_IDS / NON_VEGETATION_BROAD_IDS are empty in config — "
            "fill in the broad_id values for each group before running Step 7a."
        )
    overlap = set(VEGETATION_BROAD_IDS) & set(NON_VEGETATION_BROAD_IDS)
    if overlap:
        raise ValueError(f"broad_id values appear in both groups: {overlap}")

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
    gdf = gpd.read_file(cfg.V2_TRAINING_DATA)

    required = {"broad_habitat_type", "broad_id"}
    if not required.issubset(gdf.columns):
        raise ValueError(f"Training GeoJSON must have fields: {required}")

    if gdf.crs is None or gdf.crs != crs:
        print(f"  Reprojecting training data to {crs}...")
        gdf = gdf.to_crs(crs)

    gdf = gdf.dropna(subset=["broad_habitat_type", "broad_id"]).copy()
    gdf["broad_id"] = gdf["broad_id"].astype(int)

    if (gdf["broad_id"] < 0).any():
        raise ValueError("broad_id must be >= 0 (-1 is reserved for nodata).")

    broad_mapping = build_id_name_mapping(gdf, "broad_id", "broad_habitat_type")
    (output_dir / "class_mapping_broad.json").write_text(
        json.dumps({str(k): v for k, v in sorted(broad_mapping.items())}, indent=2),
        encoding="utf-8",
    )

    found_ids = set(gdf["broad_id"].unique())
    known_ids = set(VEGETATION_BROAD_IDS) | set(NON_VEGETATION_BROAD_IDS)
    unknown = found_ids - known_ids
    if unknown:
        print(f"  WARNING: these broad_id values are not in VEGETATION_BROAD_IDS or "
              f"NON_VEGETATION_BROAD_IDS and will be ignored: {unknown}")

    gdf = gdf[gdf["broad_id"].isin(known_ids)].copy()
    print(f"  {len(gdf)} training polygons | broad classes:")
    for code, name in sorted(broad_mapping.items()):
        grp = "VEG" if code in VEGETATION_BROAD_IDS else "NON-VEG"
        print(f"    {code}: {name:<30} [{grp}]")

    # ── Assign veg/non-veg codes ───────────────────────────────────────────────
    VEG_CODE     = 1
    NON_VEG_CODE = 2
    gdf["_veg_code"] = gdf["broad_id"].apply(
        lambda c: VEG_CODE if c in VEGETATION_BROAD_IDS else NON_VEG_CODE
    )

    VEG_NONVEG_NAMES = {VEG_CODE: "Vegetation", NON_VEG_CODE: "Non_Vegetation"}
    (output_dir / "class_mapping_veg_nonveg.json").write_text(
        json.dumps({str(k): v for k, v in VEG_NONVEG_NAMES.items()}, indent=2),
        encoding="utf-8",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Veg / Non-Veg classification
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Stage 1: Vegetation / Non-Vegetation ──────────────────────────────")

    veg_raster = rasterize_field(gdf, "_veg_code", height, width, transform)
    flat_veg   = veg_raster.ravel()
    train_mask = (flat_veg != -1) & flat_valid

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
    write_class_labels(veg_tif, VEG_NONVEG_NAMES)
    print(f"  Saved → {veg_tif.name}")
    print_pixel_counts(veg_pred_flat, VEG_NONVEG_NAMES, "Veg/Non-Veg")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Broad classification (one RF per veg group), codes = broad_id
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Stage 2: Broad classification ─────────────────────────────────────")

    broad_raster = rasterize_field(gdf, "broad_id", height, width, transform)
    flat_broad   = broad_raster.ravel()

    broad_pred_flat = np.full(flat_valid.size, -1, dtype=np.int16)

    for group_label, veg_code, id_list in [
        ("Vegetation",     VEG_CODE,     VEGETATION_BROAD_IDS),
        ("Non-Vegetation", NON_VEG_CODE, NON_VEGETATION_BROAD_IDS),
    ]:
        print(f"\n  [{group_label}]")

        # Pixels the veg classifier assigned to this group
        group_pred_mask = (veg_pred_flat == veg_code)

        group_ids_present = [i for i in id_list if i in found_ids]
        if not group_ids_present:
            print(f"  WARNING: no training polygons for {group_label} — skipping.")
            continue

        train_mask_broad = np.isin(flat_broad, group_ids_present) & flat_valid
        X_tr = X_full[train_mask_broad]
        y_tr = flat_broad[train_mask_broad]

        print(f"  Labelled pixels: {len(y_tr):,}")
        for code in sorted(group_ids_present):
            name  = broad_mapping.get(code, "?")
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

    broad_tif = output_dir / "classification_broad.tif"
    save_tif(broad_map, out_profile, broad_tif)
    write_class_labels(broad_tif, broad_mapping)
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
