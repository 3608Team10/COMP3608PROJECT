"""
Fake News Dataset Ingestion Script
Combines 3 Kaggle datasets into a single unified DataFrame


Datasets:
1. bhavikjikadara
    - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection
    - fake.csv
    - true.csv
    - (title, text, subject, date)
2. mahdimashayekhi
    - https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset
    - fake_news_dataset.csv
    - (title, text, data, source, author, category, label)
3. shawkyelgendy
    - https://www.kaggle.com/datasets/shawkyelgendy/fake-news-football
    - fake.csv
    - real.csv
    - (tweet)


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


Kaggle API key setup:
1. Google Colab Secrets: add KAGGLE_API_TOKEN as a secret
1. Google Colab Secrets: add KAGGLE_USERNAME and KAGGLE_KEY as secrets
3. Environment variable: set KAGGLE_API_TOKEN
2. Environment variables: set KAGGLE_USERNAME and KAGGLE_KEY
3. kaggle.json file at ~/.kaggle/kaggle.json


df columns: title | text | label | category | source_dataset
"""


import os
import json
# import shutil
import warnings
# import zipfile

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from google.colab import userdata # type: ignore

from pathlib import Path

warnings.filterwarnings("ignore")


# --- Constants ---

# DRIVE_ROOT = Path("/content/drive/MyDrive/project/COMP3608PROJECT")
# DATA_SUBDIR = "data"

# BHAVIK_DIR = "bhavikjikadara"
# MAHDI_DIR = "mahdimashayekhi"
# SHAWKY_DIR = "shawkyelgendy"

# Kaggle dataset slugs (owner/dataset-name)
# KAGGLE_DATASETS = {
#     BHAVIK_DIR: "bhavikjikadara/fake-news-detection",
#     MAHDI_DIR: "mahdimashayekhi/fake-news-detection-dataset",
#     SHAWKY_DIR: "shawkyelgendy/fake-news-football"
# }

BHAVIK_DIR = "bhavikjikadara/fake-news-detection"
MAHDI_DIR = "mahdimashayekhi/fake-news-detection-dataset"
SHAWKY_DIR = "shawkyelgendy/fake-news-football"


# --- Helper Functions ---

# def resolve_data_dir(mode: str, local_dir) -> Path:
#     # Return the root data directory (the one containing the three sub-folders) 
#     if mode == "drive":
#         data_dir = DRIVE_ROOT / DATA_SUBDIR
#         data_dir.mkdir(parents=True, exist_ok=True)
#         return data_dir

#     if mode == "local":
#         base = Path(local_dir) if local_dir else Path(__file__).parent
#         data_dir = base / DATA_SUBDIR
#         if not data_dir.exists():
#             data_dir = base
#         return data_dir

#     if mode == "auto":
#         drive_dir = DRIVE_ROOT / DATA_SUBDIR
#         if drive_dir.exists():
#             return drive_dir
#         script_dir = Path(__file__).parent if "__file__" in globals() else Path(".")
#         local_data = script_dir / DATA_SUBDIR
#         return local_data if local_data.exists() else script_dir

#     raise ValueError(f"Unknown mode '{mode}'. Use 'drive', 'local', or 'auto'.")


# def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
#     # Read a CSV with common encoding fallbacks
#     for enc in ("utf-8", "latin-1", "cp1252"):
#         try:
#             return pd.read_csv(path, encoding=enc, **kwargs)
#         except UnicodeDecodeError:
#             continue
#     raise UnicodeDecodeError(f"Could not decode {path} with any supported encoding.")


# def find_file(directory: Path, filename: str) -> Path | None:
#     # Case-insensitive file lookup inside a directory
#     if not directory.exists():
#         return None
#     for f in directory.iterdir():
#         if f.is_file() and f.name.lower() == filename.lower():
#             return f
#     return None


# Write credentials to ~/.kaggle/kaggle.json so kagglehub picks them up
def write_kaggle_json(data: dict):
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(json.dumps(data))
    kaggle_json.chmod(0o600)


def colab_secret(key: str):
    try:
        return userdata.get(key)
    except Exception:
        return None


# --- Kaggle Authentication & Download ---

"""
Resolve Kaggle credentials in priority order:
    1. Google Colab Secrets — KAGGLE_API_TOKEN (Token)
    2. Google Colab Secrets — KAGGLE_USERNAME + KAGGLE_KEY (Legacy)
    3. Environment variable — KAGGLE_API_TOKEN (Token)
    4. Environment variables — KAGGLE_USERNAME + KAGGLE_KEY (Legacy)
    5. ~/.kaggle/kaggle.json file (auto-detected by kagglehub)
"""
def setup_kaggle_credentials():
    # Case: Already set in environment so do nothing
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return
    
    # Case 1 & 3: KAGGLE_API_TOKEN
    api_token = colab_secret("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_API_TOKEN")
    if api_token:
        os.environ["KAGGLE_API_TOKEN"] = api_token.strip()
        print("Kaggle credentials set from KAGGLE_API_TOKEN (new OAuth token format).")
        return
    
    # # Case 2 & 5: KAGGLE_TOKEN — legacy JSON token {"username": "...", "key": "..."}
    # token_json = colab_secret("KAGGLE_TOKEN") or os.environ.get("KAGGLE_TOKEN")
    # if token_json:
    #     try:
    #         token_data = json.loads(token_json.strip())
    #         write_kaggle_json(token_data)
    #         print("Kaggle credentials written from KAGGLE_TOKEN (JSON format).")
    #         return
    #     except json.JSONDecodeError:
    #         pass  # not valid JSON; fall through
    
    # Case 2 & 4: Legacy KAGGLE_USERNAME + KAGGLE_KEY
    username = colab_secret("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    key = colab_secret("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY")
    if username and key:
        write_kaggle_json({"username": username, "key": key})
        print("Kaggle credentials written from KAGGLE_USERNAME / KAGGLE_KEY.")
        return
    
    # Case 5: Nothing found; kagglehub will raise a clear error on its own
    raise EnvironmentError(
        "Kaggle credentials not found. Provide them via one of:\n"
        "   a) Colab Secret  : KAGGLE_API_TOKEN (new OAuth token from kaggle.com/settings)\n"
        "   b) Colab Secrets : KAGGLE_USERNAME + KAGGLE_KEY (legacy API key)\n"
        "   c) Env variable  : KAGGLE_API_TOKEN\n"
        "   d) Env variables : KAGGLE_USERNAME + KAGGLE_KEY\n"
        "   e) File          : ~/.kaggle/kaggle.json\n\n"
        "To get a new token: kaggle.com → Settings → API → Create New Token"
    )


"""
Download all three kaggle datasets into data_dir/<subfolder>/.
Uses kagglehub, which supports both new OAuth tokens and legacy API keys.
Skips a dataset if its subfolder is already populated

Parameters:
data_dir : Path
    Root directory containing the three sub-folders
force : bool
    Re-download even if files already exist (default False)
"""
# def download_datasets(data_dir: Path, force: bool = False):
#     setup_kaggle_credentials()
     
#     for subdir, slug in KAGGLE_DATASETS.items():
#         dest = data_dir / subdir
#         dest.mkdir(parents=True, exist_ok=True)
        
#         # Skip if already populated and not forcing
#         existing_csvs = list(dest.glob("*.csv"))
#         if existing_csvs and not force:
#             print(f"[Kaggle] '{subdir}/' already contains {len(existing_csvs)} CSV(s) - skipping download.")
#             continue

#         print(f"[Kaggle] Downloading {slug} -> {dest} ...")
        
#         # kagglehub downloads to its own cache and returns the local path
#         cached_path = kagglehub.dataset_download(slug)
#         cached_path = Path(cached_path)

#         # Copy / extract every CSV from the cached location into dest
#         collect_csvs(cached_path, dest)

#         print(f"[Kaggle] Done: {subdir}/")

"""
Recursively copy all .csv files from src into dest (flat — no sub-dirs).
Also extracts .csv files from any .zip archives found.
"""
# def collect_csvs(src: Path, dest: Path):
#     if src.is_file():
#         if src.suffix.lower() == ".csv":
#             shutil.copy2(src, dest / src.name)
#         elif src.suffix.lower() == ".zip":
#             extract_csvs_from_zip(src, dest)
#         return
    
#     for item in src.rglob("*"):
#         if item.is_file():
#             if item.suffix.lower() == ".csv":
#                 shutil.copy2(item, dest / item.name)
#             elif item.suffix.lower() == ".zip":
#                 extract_csvs_from_zip(item, dest)


# def extract_csvs_from_zip(zip_path: Path, dest: Path):
#     print(f"[Kaggle] Extracting {zip_path.name} ...")
#     with zipfile.ZipFile(zip_path, "r") as zf:
#         for member in zf.namelist():
#             if member.lower().endswith(".csv"):
#                 target = dest / Path(member).name
#                 with zf.open(member) as src_f, open(target, "wb") as dst_f:
#                     dst_f.write(src_f.read())


# --- Dataset Loaders ---

# def load_bhavik(data_dir: Path) -> pd.DataFrame:
#     # fake.csv -> label = 0, true.csv -> label = 1
    
#     src_dir = data_dir / BHAVIK_DIR
#     frames = []

#     for fname, lbl in [("fake.csv", 0), ("true.csv", 1)]:
#         path = find_file(src_dir, fname)
#         if path is None:
#             print(f"[bhavik] WARNING: '{BHAVIK_DIR}/{fname}' not found - skipping.")
#             continue
#         df = read_csv_safe(path)
        
#         cols = {"title": "", "text": "", "subject": "Unknown"}
#         for col, default in cols.items():
#             if col not in df.columns:
#                 df[col] = default
        
#         df = df[["title", "text", "subject"]].copy()
#         df = df.rename(columns={"subject": "category"})
#         df["label"] = lbl
#         df["source_dataset"] = "bhavikjikadara"
#         frames.append(df)
#         print(f"[bhavik] Loaded '{fname}': {len(df):,} rows (label={lbl})")

#     if not frames:
#         return pd.DataFrame(columns=["title", "text", "label", "category", "source_dataset"])
#     return pd.concat(frames, ignore_index=True)


def load_df(dir: str, path: str) -> pd.DataFrame:
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dir,
        path
    )


def load_bhavik() -> pd.DataFrame:
    df_true = load_df(BHAVIK_DIR, "true.csv")
    df_fake = load_df(BHAVIK_DIR, "fake.csv")
    
    df_true['label'] = 1
    df_fake['label'] = 0
    
    df = pd.concat([df_true, df_fake], ignore_index=True)
    
    df = df.drop(columns=['date'])
    df = df.rename(columns={'subject': 'category'})
    df['source_dataset'] = 'bhavikjikadara'
    
    print(df.head())
    return df


# def load_mahdi(data_dir: Path) -> pd.DataFrame:
#     # 'label' column: 'fake' -> 0, 'real' -> 1
    
#     fname = "fake_news_dataset.csv"
#     path = find_file(data_dir / MAHDI_DIR, fname)

#     if path is None:
#         print(f"[mahdi] WARNING: '{MAHDI_DIR}/{fname}' not found - skipping.")
#         return pd.DataFrame(columns=["title", "text", "label", "category", "source_dataset"])

#     df = read_csv_safe(path)

#     df["label"] = (
#         df["label"]
#         .astype(str).str.strip().str.lower()
#         .map({"fake": 0, "real": 1, "0": 0, "1": 1})
#     )
#     unmapped = df["label"].isna().sum()
#     if unmapped:
#         print(f"[mahdi] WARNING: {unmapped} rows had unrecognised labels - dropped.")
#         df = df.dropna(subset=["label"])

#     df["label"] = df["label"].astype(int)
    
#     if "category" not in df.columns:
#         df["category"] = "Unknown"
    
#     df = df[["title", "text", "label", "category"]].copy()
#     df["source_dataset"] = "mahdimashayekhi"
#     print(f"[mahdi] Loaded '{fname}': {len(df):,} rows")
#     return df


def load_mahdi() -> pd.DataFrame:
    df = load_df(MAHDI_DIR, "fake_news_dataset.csv")
    
    print(df.head())
    return df


# def load_shawky(data_dir: Path) -> pd.DataFrame:
#     # fake.csv -> label = 0, real.csv -> label = 1
#     # Column 'tweet' is mapped to 'text'; title is set to ''.
    
#     src_dir = data_dir / SHAWKY_DIR
#     frames = []

#     for fname, lbl in [("fake.csv", 0), ("real.csv", 1)]:
#         path = find_file(src_dir, fname)
#         if path is None:
#             print(f"[shawky] WARNING: '{SHAWKY_DIR}/{fname}' not found - skipping.")
#             continue
#         df = read_csv_safe(path)

#         if "tweet" in df.columns:
#             df = df[["tweet"]].rename(columns={"tweet": "text"})
#         elif "text" in df.columns:
#             df = df[["text"]].copy()
#         else:
#             df = df.iloc[:, [0]].copy()
#             df.columns = ["text"]

#         df["title"] = ""
#         df["label"] = lbl
#         df["category"] = "Unknown"
#         df["source_dataset"] = "shawkyelgendy"
#         frames.append(df)
#         print(f"[shawky] Loaded '{fname}': {len(df):,} rows (label={lbl})")

#     if not frames:
#         return pd.DataFrame(columns=["title", "text", "label", "category", "source_dataset"])
#     return pd.concat(frames, ignore_index=True)


def load_shawky() -> pd.DataFrame:
    # Set the path to the file you'd like to load
    filepath = ""
    
    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        SHAWKY_DIR,
        filepath
    )
    
    print(df.head())
    return df


# --- Main Loader ---

"""
Download (if needed) and combine all three fake-news datasets into one DataFrame.

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
force_download : bool
    Re-download datasets even if local CSVs already exist. Default False.

Returns:

pd.DataFrame with columns:
    title          - article/tweet title ('' for tweet-only rows)
    text           - main text content
    label          - 0 = fake, 1 = real
    category       - topic category ('Unknown' where unavailable)
    source_dataset - origin dataset name
"""
# def load_all_datasets(
#     mode: str = "auto",
#     local_dir=None,
#     drop_duplicates: bool = True,
#     drop_na_text: bool = True,
#     force_download: bool = False
# ) -> pd.DataFrame:
#     print("-" * 60)
#     print("Fake News Dataset Ingestion")
#     print("-" * 60)

#     data_dir = resolve_data_dir(mode, local_dir)
#     print(f"\nData directory : {data_dir}")
#     print(f"Sub-folders: {BHAVIK_DIR}/  {MAHDI_DIR}/  {SHAWKY_DIR}/\n")
    
#     # Download from Kaggle if needed
#     print("Checking / downloading datasets from Kaggle ...")
#     download_datasets(data_dir, force=force_download)
#     print()

#     print("Loading bhavikjikadara ...")
#     df_bhavik = load_bhavik(data_dir)

#     print("\nLoading mahdimashayekhi ...")
#     df_mahdi = load_mahdi(data_dir)

#     print("\nLoading shawkyelgendy ...")
#     df_shawky = load_shawky(data_dir)

#     # Combine
#     combined = pd.concat([df_bhavik, df_mahdi, df_shawky], ignore_index=True)
#     combined = combined[["title", "text", "label", "category", "source_dataset"]]
    
#     # Normalise category values
#     combined["category"] = combined["category"].fillna("Unknown").astype(str).str.strip()
#     combined.loc[combined["category"] == "", "category"] = "Unknown"

#     # Basic cleaning
#     if drop_na_text:
#         before = len(combined)
#         combined = combined[combined["text"].notna() & (combined["text"].str.strip() != "")]
#         dropped = before - len(combined)
#         if dropped:
#             print(f"\nDropped {dropped:,} rows with empty/null text.")

#     if drop_duplicates:
#         before = len(combined)
#         combined = combined.drop_duplicates(subset=["text", "label"])
#         dropped = before - len(combined)
#         if dropped:
#             print(f"Dropped {dropped:,} duplicate rows.")

#     combined = combined.reset_index(drop=True)

#     # Summary
#     print("\n" + "-" * 60)
#     print("Combined Dataset Summary")
#     print("-" * 60)
#     print(f"Total rows: {len(combined):,}")
#     print(f"Fake (0): {(combined['label'] == 0).sum():,}")
#     print(f"Real (1): {(combined['label'] == 1).sum():,}")
#     print(f"\nRows per source:")
#     for src, count in combined["source_dataset"].value_counts().items():
#         print(f"{src:<22} {count:,}")
#     print(f"\nTop categories:")
#     for cat, count in combined["category"].value_counts().head(10).items():
#         print(f"  {cat:<22} {count:,}")
#     print("-" * 60)

#     return combined


# --- Data Persistence ---

# def save_combined(
#     df: pd.DataFrame,
#     filename: str = "merged_fake_news.csv",
#     mode: str = "auto",
#     local_dir=None,
# ) -> Path:
#     # Save the combined DataFrame to CSV inside the data directory
#     data_dir = resolve_data_dir(mode, local_dir)
#     out_path = data_dir / filename
#     df.to_csv(out_path, index=False)
#     print(f"Saved -> {out_path}")
#     return out_path


# --- CLI Test ---

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Ingest fake-news datasets.")
#     parser.add_argument("--mode", choices=["drive", "local", "auto"], default="auto")
#     parser.add_argument("--data-dir", default=None, help="Base directory for CSVs")
#     parser.add_argument("--save", action="store_true", help="Save combined CSV")
#     parser.add_argument("--force-download", action="store_true",
#         help="Re-download datasets even if already present")
#     args = parser.parse_args()

#     df = load_all_datasets(
#         mode=args.mode, 
#         local_dir=args.data_dir,
#         force_download=args.force_download
#     )
#     print("\nFirst 5 rows:")
#     print(df.head(5).to_string())

#     if args.save:
#         save_combined(df, mode=args.mode, local_dir=args.data_dir)

