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



