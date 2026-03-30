"""
Fake News Dataset Ingestion Script
Combines 3 Kaggle datasets into a single unified DataFrame


Datasets:
1. bhavikjikadara
    - fake.csv
    - true.csv
    - (title, text, subject, date)
2. mahdimashayekhi
    - fake_news_dataset.csv
    - (title, text, data, source, author, category, label)
3. shawkyelgendy
    - fake.csv
    - real.csv
    - (tweet)


Expected folder structure under your data directory:
data/
├── bhavikjikadara/
│   ├── fake.csv
│   └── true.csv
├── mahdimashayekhi/
│   └── fake_news_dataset.csv
└── shawkyelgendy/
    ├── fake.csv
    └── real.csv


Compatible with:
- Google Colab (Drive mounted at /content/drive)
- Standalone upload (script uploaded at Colab session)


Usage:
1. Google Colab with mounted Drive:
from data_ingestion import load_all_datasets
df = load_all_datasets(mode="drive")

2. Standalone script upload to Colab session:
from data_ingestion import load_all_datasets
df = load_all_datasets(mode="local", local_dir="/content")

3. Auto-detect mode (tries Drive first, then local):
from data_ingestion import load_all_datasets
df = load_all_datasets()


df columns: title | text | label | source_dataset
"""


import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


# --- Constants ---

DRIVE_ROOT = Path("/content/drive/MyDrive/project/COMP3608PROJECT")
DATA_SUBDIR = "data"

BHAVIK_DIR = "bhavikjikadara"
MAHDI_DIR = "mahdimashayekhi"
SHAWKY_DIR = "shawkyelgendy"


# --- Helper Functions ---

def resolve_data_dir(mode: str, local_dir) -> Path:
    # Return the root data directory (the one containing the three sub-folders) 
    if mode == "drive":
        data_dir = DRIVE_ROOT / DATA_SUBDIR
        if not data_dir.exists():
            scaffold_data_dir(data_dir)
            raise FileNotFoundError(
                f"Data directory scaffolded at:\n{data_dir}\n"
                "Please upload your CSVs into the sub-folders shown and re-run."
            )
        return data_dir

    if mode == "local":
        base = Path(local_dir) if local_dir else Path(__file__).parent
        data_dir = base / DATA_SUBDIR
        # Also accept passing the data dir itself directly
        if not data_dir.exists():
            data_dir = base
        return data_dir

    if mode == "auto":
        drive_dir = DRIVE_ROOT / DATA_SUBDIR
        if drive_dir.exists():
            return drive_dir
        # Fall back to a 'data/' folder next to this script, or the script dir
        script_dir = Path(__file__).parent if "__file__" in globals() else Path(".")
        local_data = script_dir / DATA_SUBDIR
        return local_data if local_data.exists() else script_dir

    raise ValueError(f"Unknown mode '{mode}'. Use 'drive', 'local', or 'auto'.")


def scaffold_data_dir(data_dir: Path):
    # Create the expected sub-folder structure so the user knows where to put files
    for sub in (BHAVIK_DIR, MAHDI_DIR, SHAWKY_DIR):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    print(f"Created folder structure under: {data_dir}")


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    # Read a CSV with common encoding fallbacks
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {path} with any supported encoding.")


def find_file(directory: Path, filename: str) -> Path | None:
    # Case-insensitive file lookup inside a directory
    if not directory.exists():
        return None
    for f in directory.iterdir():
        if f.is_file() and f.name.lower() == filename.lower():
            return f
    return None




