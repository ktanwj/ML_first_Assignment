# import libraries
import os
from typing import Dict, Any
import toml

# path locations
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_PATH, "models")
DATA_PATH = os.path.join(PROJECT_PATH, "data")
FILE_PATH = os.path.join(DATA_PATH, "fishing.db")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "output")
PREDICTION_PATH = os.path.join(OUTPUT_PATH, "model prediction")
METRICS_PATH = os.path.join(OUTPUT_PATH, "model performance")

# data related configs
COLS_TO_DROP = ['ColourOfBoats', 'WindDir3pm', 'WindSpeed3pm', 'WindDir9am', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm'] # indicates which columns to drop 
COLS_TO_UPPERCASE = ['Pressure9am'] # indicates which columns to apply uppercase treatment
COLS_TO_OH = ['Location', 'Pressure9am']
HYPERPARAMETER_TUNING = False

# To read model config file in toml format
def read_model_config() -> Dict[str, Any]:
    filepath = os.path.join(PROJECT_PATH, "model_config.toml")
    with open(filepath, "r", encoding="utf-8") as f:
        return toml.load(f)