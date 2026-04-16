# ── Chelmsford Phase 2 — project configuration ────────────────────────────────

INPUT_DIR  = r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\Sentinel-2 pre-processed cloud_mask"
OUTPUT_DIR = r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\Sentinel-2 pre-processed cloud_mask_clip_to_aoi"
AOI        = r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\catchment_data\AOI_project_AOI_merged_with_500m_buffer_EPSG_32630.geojson"


CLOUD_MASK_INPUT =r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\Sentinel-2 pre-processed cloud_mask_clip_to_aoi"
CLOUD_MASK_OUTPUT = r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\Sentinel-2 pre-processed cloud_mask_clip_to_aoi_MASKED"


INDICIES_INPUT = CLOUD_MASK_OUTPUT
INDICIES_OUTPUT = r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\Sentinel-2 pre-processed cloud_mask_clip_to_aoi_MASKED_INDICIES"

DATA_CUBE_INPUT  = INDICIES_OUTPUT
DATA_CUBE_OUTPUT = r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\data_cube\chelmsford_phase2_datacube.nc"

STATS_INPUT  = DATA_CUBE_OUTPUT
STATS_OUTPUT = r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\data_cube\chelmsford_phase2_statistics.tif"

CLASSIFICATION_INPUT      = STATS_OUTPUT
CLASSIFICATION_OUTPUT_DIR = r"C:\OneDrive\Envance\Envance Ltd - Documents\10 R&D\chelmsford\Phase 2\data\classification"
ELBOW_K_MIN = 5
ELBOW_K_MAX = 30
