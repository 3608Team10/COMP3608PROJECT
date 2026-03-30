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


# --- Dataset Loaders ---

def load_bhavik(data_dir: Path) -> pd.DataFrame:
    # fake.csv -> label = 0, true.csv -> label = 1
    
    src_dir = data_dir / BHAVIK_DIR
    frames = []

    for fname, lbl in [("fake.csv", 0), ("true.csv", 1)]:
        path = find_file(src_dir, fname)
        if path is None:
            print(f"[bhavik] WARNING: '{BHAVIK_DIR}/{fname}' not found - skipping.")
            continue
        df = read_csv_safe(path)
        df = df[["title", "text"]].copy()
        df["label"] = lbl
        df["source_dataset"] = "bhavikjikadara"
        frames.append(df)
        print(f"[bhavik] Loaded '{fname}': {len(df):,} rows (label={lbl})")

    if not frames:
        return pd.DataFrame(columns=["title", "text", "label", "source_dataset"])
    return pd.concat(frames, ignore_index=True)


def load_mahdi(data_dir: Path) -> pd.DataFrame:
    # 'label' column: 'fake' -> 0, 'real' -> 1
    
    fname = "fake_news_dataset.csv"
    path = find_file(data_dir / MAHDI_DIR, fname)

    if path is None:
        print(f"[mahdi] WARNING: '{MAHDI_DIR}/{fname}' not found - skipping.")
        return pd.DataFrame(columns=["title", "text", "label", "source_dataset"])

    df = read_csv_safe(path)

    df["label"] = (
        df["label"]
        .astype(str).str.strip().str.lower()
        .map({"fake": 0, "real": 1, "0": 0, "1": 1})
    )
    unmapped = df["label"].isna().sum()
    if unmapped:
        print(f"[mahdi] WARNING: {unmapped} rows had unrecognised labels - dropped.")
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)
    df = df[["title", "text", "label"]].copy()
    df["source_dataset"] = "mahdimashayekhi"
    print(f"[mahdi] Loaded '{fname}': {len(df):,} rows")
    return df


def load_shawky(data_dir: Path) -> pd.DataFrame:
    # fake.csv -> label = 0, real.csv -> label = 1
    # Column 'tweet' is mapped to 'text'; title is set to ''.
    
    src_dir = data_dir / SHAWKY_DIR
    frames = []

    for fname, lbl in [("fake.csv", 0), ("real.csv", 1)]:
        path = find_file(src_dir, fname)
        if path is None:
            print(f"[shawky] WARNING: '{SHAWKY_DIR}/{fname}' not found - skipping.")
            continue
        df = read_csv_safe(path)

        # Map whichever column holds the text
        if "tweet" in df.columns:
            df = df[["tweet"]].rename(columns={"tweet": "text"})
        elif "text" in df.columns:
            df = df[["text"]].copy()
        else:
            df = df.iloc[:, [0]].copy()
            df.columns = ["text"]

        df["title"] = ""
        df["label"] = lbl
        df["source_dataset"] = "shawkyelgendy"
        frames.append(df)
        print(f"[shawky] Loaded '{fname}': {len(df):,} rows (label={lbl})")

    if not frames:
        return pd.DataFrame(columns=["title", "text", "label", "source_dataset"])
    return pd.concat(frames, ignore_index=True)


# --- Main Loader ---

"""
Load and combine all three fake-news datasets into one DataFrame.

Parameters:

mode : str
    'drive' - read from mounted Google Drive
    'local' - read from local_dir/data/
    'auto' - try Drive first, fall back to local (default)
local_dir : str | Path | None
    Base path when mode='local'. Defaults to the script's directory.
drop_duplicates : bool
    Remove exact duplicate (text, label) pairs.  Default True.
drop_na_text : bool
    Drop rows where 'text' is NaN or empty.  Default True.

Returns:

pd.DataFrame with columns:
    title          - article/tweet title ('' for tweet-only rows)
    text           - main text content
    label          - 0 = fake, 1 = real
    source_dataset - origin dataset name
"""
def load_all_datasets(
    mode: str = "auto",
    local_dir=None,
    drop_duplicates: bool = True,
    drop_na_text: bool = True,
) -> pd.DataFrame:
    print("-" * 60)
    print("  Fake News Dataset Ingestion")
    print("-" * 60)

    data_dir = resolve_data_dir(mode, local_dir)
    print(f"\nData directory : {data_dir}")
    print(f"Sub-folders: {BHAVIK_DIR}/  {MAHDI_DIR}/  {SHAWKY_DIR}/\n")

    print("Loading bhavikjikadara ...")
    df_bhavik = load_bhavik(data_dir)

    print("\nLoading mahdimashayekhi ...")
    df_mahdi = load_mahdi(data_dir)

    print("\nLoading shawkyelgendy ...")
    df_shawky = load_shawky(data_dir)

    # Combine
    combined = pd.concat([df_bhavik, df_mahdi, df_shawky], ignore_index=True)
    combined = combined[["title", "text", "label", "source_dataset"]]

    # Basic cleaning
    if drop_na_text:
        before = len(combined)
        combined = combined[combined["text"].notna() & (combined["text"].str.strip() != "")]
        dropped = before - len(combined)
        if dropped:
            print(f"\nDropped {dropped:,} rows with empty/null text.")

    if drop_duplicates:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["text", "label"])
        dropped = before - len(combined)
        if dropped:
            print(f"Dropped {dropped:,} duplicate rows.")

    combined = combined.reset_index(drop=True)

    # Summary
    print("\n" + "-" * 60)
    print("Combined Dataset Summary")
    print("-" * 60)
    print(f"Total rows: {len(combined):,}")
    print(f"Fake (0): {(combined['label'] == 0).sum():,}")
    print(f"Real (1): {(combined['label'] == 1).sum():,}")
    print(f"\nRows per source:")
    for src, count in combined["source_dataset"].value_counts().items():
        print(f"{src:<22} {count:,}")
    print("-" * 60)

    return combined


# --- Data Persistence ---

def save_combined(
    df: pd.DataFrame,
    filename: str = "merged_fake_news.csv",
    mode: str = "auto",
    local_dir=None,
) -> Path:
    # Save the combined DataFrame to CSV inside the data directory
    data_dir = resolve_data_dir(mode, local_dir)
    out_path = data_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return out_path

