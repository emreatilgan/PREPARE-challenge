import os
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Data file paths
TRAIN_FEATURES_PATH = RAW_DATA_DIR / "train_features.csv"
TEST_FEATURES_PATH = RAW_DATA_DIR / "test_features.csv"
TRAIN_LABELS_PATH = RAW_DATA_DIR / "train_labels.csv"
SUBMISSION_FORMAT_PATH = RAW_DATA_DIR / "submission_format.csv"

# Model parameters
RANDOM_STATE = 42
N_FOLDS = 5

# Feature groups
TEMPORAL_FEATURES = [col for col in ['_03', '_12'] if col in train_features.columns]
DEMOGRAPHIC_FEATURES = ['age', 'ragender', 'urban', 'married', 'edu_gru']
HEALTH_FEATURES = ['glob_hlth', 'n_adl', 'n_iadl', 'n_depr', 'n_illnesses']
BEHAVIORAL_FEATURES = ['exer_3xwk', 'alcohol', 'tobacco']
SOCIOECONOMIC_FEATURES = ['employment', 'hincome', 'insured']