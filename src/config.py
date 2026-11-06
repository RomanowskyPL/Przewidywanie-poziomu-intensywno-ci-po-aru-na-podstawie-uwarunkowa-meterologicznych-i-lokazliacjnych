import os

# Ścieżki do plików
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'wildfire_multi_region_dataset_missing.csv')

# Zmienne konfiguracyjne
TARGET_COL = 'fire_intensity'
DATE_COL = 'acq_date'
DROP_COLS = [DATE_COL] 

# Parametry globalne
RANDOM_STATE = 42
TEST_SIZE = 0.2
SEEDS_TO_TEST = [10, 42, 123, 999, 2026]
TRAIN_SIZES = [0.2, 0.4, 0.6, 0.8]