"""Rutas canónicas del proyecto GameStrategist."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASET_PATH = DATA_DIR / "steam_game_hits.csv"

MODULE_1_DIR = ARTIFACTS_DIR / "module_1"
MODULE_1_MODEL_PATH = MODULE_1_DIR / "random_forest_optimized.joblib"
MODULE_1_CONFIG_PATH = MODULE_1_DIR / "config.json"

MODULE_2_DIR = ARTIFACTS_DIR / "module_2"
EMBEDDINGS_PATH = MODULE_2_DIR / "embeddings_e5_base.npy"
EMBEDDING_IDS_PATH = MODULE_2_DIR / "embedding_game_ids.npy"
EMBEDDING_CONFIG_PATH = MODULE_2_DIR / "embedding_config.json"
CHROMA_DIR = MODULE_2_DIR / "chroma_db"

FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"
