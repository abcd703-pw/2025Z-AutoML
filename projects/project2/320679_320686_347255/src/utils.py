"""Utility functions for model handling and data loading."""

import importlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_FILE_X, DATA_FILE_Y, TARGET_COLUMN_NAME, TEST_SIZE, RANDOM_STATE, MODELS_CONFIG, DATA_FILE

def get_model_class(config):
    """Dynamically import a class from a string."""
    class_path = config['class']
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def create_model_instance(config):
    """Create an instance of a model given its config."""
    model_class = get_model_class(config)
    params = config.get('params', {})
    return model_class(**params)

def load_models_config(file_path=MODELS_CONFIG):
    """Load a JSON models configuration file."""
    with open(file_path, 'r') as f:
        return json.load(f)
    
def load_data_one_file(file_path=DATA_FILE):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN_NAME, axis=1)
    y = data[TARGET_COLUMN_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test

def load_data(X_file_path=DATA_FILE_X, y_file_path=DATA_FILE_Y):
    """Load dataset from separate CSV files."""
    X = pd.read_csv(X_file_path)
    y = pd.read_csv(y_file_path)[TARGET_COLUMN_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test