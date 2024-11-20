import pandas as pd
from src.config import *

def load_train_data():
    """Load training data."""
    train_features = pd.read_csv(TRAIN_FEATURES_PATH)
    train_labels = pd.read_csv(TRAIN_LABELS_PATH)
    return train_features, train_labels

def load_test_data():
    """Load test data."""
    test_features = pd.read_csv(TEST_FEATURES_PATH)
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    return test_features, submission_format