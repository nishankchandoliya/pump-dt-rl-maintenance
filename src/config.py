from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

# Seed ranges are fixed before any experiment so that tuning and final testing never share episodes.
# Training episode k uses seed TRAIN_SEED_START + k.
VAL_SEEDS = list(range(1000, 1050))
TEST_SEEDS = list(range(2000, 2200))
TRAIN_SEED_START = 10_000
