"""
Step 6 — Unsupervised K-means classification with elbow method.

Phase 1 — Elbow analysis:
    Runs K-means for k = ELBOW_K_MIN to ELBOW_K_MAX.
    Saves an elbow plot so you can visually pick the best k values.

Phase 2 — Classification:
    Prompts you to enter the k values you want to classify (comma-separated).
    Exports one GeoTIFF per k value, with each pixel labelled by cluster.

Input: statistics GeoTIFF from Step 5 (102 bands).

Reads paths from Project/Chelmsford_phase_2_config.py.
"""

import importlib.util
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Load config
_config_path = Path(__file__).parent.parent / "Project" / "Chelmsford_phase_2_config.py"
_spec = importlib.util.spec_from_file_location("cfg", _config_path)
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)


def load_features(input_path):
    """
    Load the statistics GeoTIFF and return:
        X           — (n_valid_pixels, n_features) array for clustering
        valid_mask  — (height, width) bool array, True where pixel is valid
        profile     — rasterio profile for writing output
    """
    with rasterio.open(input_path) as src:
        data    = src.read().astype("float32")  # (bands, height, width)
        profile = src.profile.copy()
        nodata  = src.nodata

    n_bands, height, width = data.shape

    # Pixel is valid if it has no NaN and no nodata across ALL bands
    if nodata is not None and not np.isnan(float(nodata)):
        invalid = np.any((data == nodata) | np.isnan(data), axis=0)
    else:
        invalid = np.any(np.isnan(data), axis=0)

    valid_mask = ~invalid  # (height, width)

    # Flatten to (n_valid_pixels, n_features)
    pixels = data[:, valid_mask].T  # (n_valid, n_bands)

    return pixels, valid_mask, height, width, profile


def run_elbow(X, k_min, k_max, output_dir):
    """Run K-means for k_min..k_max, plot inertia, save the plot."""
    print(f"\nRunning elbow analysis (k={k_min} to k={k_max})...")
    ks       = range(k_min, k_max + 1)
    inertias = []

    for k in ks:
        print(f"  k={k} ...", end=" ", flush=True)
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"inertia={km.inertia_:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(ks), inertias, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (within-cluster sum of squares)", fontsize=12)
    ax.set_title("Elbow Method — Optimal k for Habitat Classification", fontsize=13)
    ax.set_xticks(list(ks))
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_path = output_dir / "elbow_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\nElbow plot saved → {plot_path}")

    return list(ks), inertias


def classify(X, valid_mask, height, width, profile, k, output_dir):
    """Run K-means with k clusters and save the classified GeoTIFF."""
    print(f"  Classifying k={k} ...", end=" ", flush=True)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X).astype("int16") + 1  # 1-based class labels

    # Reconstruct full raster (nodata=0 for invalid pixels)
    result = np.zeros((height, width), dtype="int16")
    result[valid_mask] = labels

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="int16", nodata=0)

    out_path = output_dir / f"classification_k{k:02d}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(result[np.newaxis, ...])

    print(f"saved → {out_path.name}")


def main():
    input_path  = Path(cfg.CLASSIFICATION_INPUT)
    output_dir  = Path(cfg.CLASSIFICATION_OUTPUT_DIR)
    k_min       = cfg.ELBOW_K_MIN
    k_max       = cfg.ELBOW_K_MAX
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and scale features ────────────────────────────────────────────────
    print(f"Loading features: {input_path.name}")
    X, valid_mask, height, width, profile = load_features(input_path)
    print(f"  Valid pixels: {X.shape[0]:,}  |  Features: {X.shape[1]}")

    print("Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Phase 1: Elbow ─────────────────────────────────────────────────────────
    run_elbow(X_scaled, k_min, k_max, output_dir)

    # ── Phase 2: Choose k values and classify ─────────────────────────────────
    print("\nLook at the elbow plot and choose which k values to classify.")
    raw = input("Enter k values separated by commas (e.g. 8,10,12,15): ").strip()

    try:
        k_values = [int(k.strip()) for k in raw.split(",") if k.strip()]
    except ValueError:
        print("Invalid input — please enter integers separated by commas.")
        return

    if not k_values:
        print("No k values entered. Exiting.")
        return

    print(f"\nClassifying for k = {k_values} ...")
    for k in k_values:
        classify(X_scaled, valid_mask, height, width, profile, k, output_dir)

    print(f"\nDone. {len(k_values)} classification(s) saved → {output_dir}")


if __name__ == "__main__":
    main()
