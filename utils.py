import os
from pathlib import Path

import yaml
#from dotenv import load_dotenv

# Load external config
PROJECT_ROOT = Path(__file__).resolve().parent
print(f"Project root: {PROJECT_ROOT}")

# Load ENV from .env file (if present) or system env
#load_dotenv(PROJECT_ROOT / ".env")

ENV = os.getenv("ENV", "dev")  # default to dev if not set


CONFIG_PATH = PROJECT_ROOT / f"config/{ENV}.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

print(f"Loaded config from {CONFIG_PATH}")
print(CONFIG)


DATA_DIR = (PROJECT_ROOT / CONFIG["data_dir"]).resolve()
print(f"Data dir: {DATA_DIR}")
BRACKET_BLACK_DATA_DIR = (PROJECT_ROOT / CONFIG["bracket_black_data_dir"]).resolve()
print(f"Bracket black data dir: {BRACKET_BLACK_DATA_DIR}")
PROCESSED_DATA_DIR = (PROJECT_ROOT / CONFIG["processed_data_dir"]).resolve()
PROCESSED_BRACKET_BLACK_DATA_DIR = (PROJECT_ROOT / CONFIG["processed_bracket_black_data_dir"]).resolve()
print(f"Processed bracket black data dir: {PROCESSED_BRACKET_BLACK_DATA_DIR}")
