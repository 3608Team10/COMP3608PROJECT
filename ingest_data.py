"""
Fake News Dataset Ingestion Script
Combines 3 Kaggle datasets into a single unified DataFrame


Datasets:
1. bhavikjikadara
    - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection
    - fake.csv | label=0
    - true.csv | label=1
    - (title, text, subject, date)
    - category is mapped from 'subject'
    - 'date' is dropped
2. mahdimashayekhi
    - https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset
    - fake_news_dataset.csv
    - (title, text, date, source, author, category, label)
    - label is mapped from 'fake'/'real' to 0/1
    - 'date', 'source', 'author' are dropped
3. shawkyelgendy
    - https://www.kaggle.com/datasets/shawkyelgendy/fake-news-football
    - fake.csv | label=0
    - real.csv | label=1
    - (tweet)
    - category is set to "Sports" for all rows
    - 'text' is mapped from 'tweet'
    - 'title' is set to ''


Compatibility/Usage:
1. Google Colab (Drive mounted at /content/drive)
2. Standalone script upload to Colab session

from ingest_data import load_datasets
df = load_datasets()


df columns: title | text | label | category | dataset
"""


import warnings
import pandas as pd

from pathlib import Path

warnings.filterwarnings("ignore")


# --- Constants ---

DRIVE_ROOT = Path("/content/drive/MyDrive/project/COMP3608PROJECT")

BHAVIK = "bhavikjikadara"
MAHDI = "mahdimashayekhi"
SHAWKY = "shawkyelgendy"

GITHUB_BASE = "https://raw.githubusercontent.com/3608Team10/COMP3608PROJECT/refs/heads/main/data"

BHAVIK_URL_FAKE = f"{GITHUB_BASE}/{BHAVIK}/fake.csv"
BHAVIK_URL_TRUE = f"{GITHUB_BASE}/{BHAVIK}/true.csv"
MAHDI_URL = f"{GITHUB_BASE}/{MAHDI}/fake_news_dataset.csv"
SHAWKY_URL_FAKE = f"{GITHUB_BASE}/{SHAWKY}/fake.csv"
SHAWKY_URL_TRUE = f"{GITHUB_BASE}/{SHAWKY}/real.csv"


# --- Helper Functions ---

def preprocess_drop_na_text(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    dropped = before - len(df)
    if dropped:
        print(f"\nDropped {dropped:,} rows with empty/null text.")
    return df


def preprocess_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} duplicate rows.")
    return df


VALID_CATEGORIES = {
    "Sports", "Politics", "News", "Health", "Entertainment", 
    "Technology", "Business", "Science", "Unknown"
}

CATEGORY_MAP = {
    "Politicsnews": "Politics",
    "Politics": "Politics",
    "Left-News": "Politics",
    "Government News": "Politics",
    "Us_News": "News",
    "Worldnews": "News",
    "News": "News",
    "Middle-East": "News",
}


def preprocess_normalise_category(df: pd.DataFrame) -> pd.DataFrame:
    df["category"] = (
        df["category"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.title()
            .replace(CATEGORY_MAP)
    )
    
    # Guard: Empty strings after title-casing become "Unknown"
    df.loc[df["category"] == "", "category"] = "Unknown"
    
    # Raise if any unexpected categories remain (should be none after mapping)
    unexpected = set(df["category"].unique()) - VALID_CATEGORIES
    if unexpected:
        raise ValueError(
            f"Unexpected categories found after normalization: {sorted(unexpected)}"
        )
    
    return df


def summarize_datasets(df: pd.DataFrame):
    print("\n" + "-" * 60)
    print("Fake News Dataset Summary")
    print("-" * 60)
    
    print(f"Total rows: {len(df):,}")
    print(f"Fake (0): {(df['label'] == 0).sum():,}")
    print(f"Real (1): {(df['label'] == 1).sum():,}")
    
    print(f"\nRows per source:")
    for src, count in df["dataset"].value_counts().items():
        print(f"{src:<22} {count:,}")
    
    print(f"\nCategories:")
    for cat, count in df["category"].value_counts().items():
        print(f"  {cat:<22} {count:,}")
    
    print("-" * 60)


# --- Dataset Loaders ---

def load_df(dir: str, path: str) -> pd.DataFrame:
    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dir,
        path
    )


def load_bhavik() -> pd.DataFrame:
    try:
        df_true = load_df(BHAVIK_DIR, "true.csv")
        print(f"[bhavik] Loaded 'true.csv': {len(df_true)} rows")
    except Exception as e:
        print(f"[bhavik] WARNING: Could not load 'true.csv' - {e}")
        df_true = pd.DataFrame(columns=["title", "text", "category"])
    
    try:
        df_fake = load_df(BHAVIK_DIR, "fake.csv")
        print(f"[bhavik] Loaded 'fake.csv': {len(df_fake)} rows")
    except Exception as e:
        print(f"[bhavik] WARNING: Could not load 'fake.csv' - {e}")
        df_fake = pd.DataFrame(columns=["title", "text", "category"])
    
    df_true['label'] = 1
    df_fake['label'] = 0
    
    df = pd.concat([df_true, df_fake], ignore_index=True)
    
    df = df.drop(columns=['date'])
    df = df.rename(columns={'subject': 'category'})
    df['dataset'] = BHAVIK

    return df


def load_mahdi() -> pd.DataFrame:
    try:
        df = load_df(MAHDI_DIR, "fake_news_dataset.csv")
        print(f"[mahdi] Loaded 'fake_news_dataset.csv': {len(df)} rows")
    except Exception as e:
        print(f"[mahdi] WARNING: Could not load 'fake_news_dataset.csv' - {e}")
        return pd.DataFrame(columns=["title", "text", "label", "category", "dataset"])
    
    df['label'] = df['label'].map({'fake': 0, 'real': 1})
    df = df.drop(columns=['date', 'source', 'author'])
    df['dataset'] = MAHDI
    
    return df


def load_shawky() -> pd.DataFrame:
    try:
        df_real = load_df(SHAWKY_DIR, "real.csv")
        df_real = df_real.rename(columns={"tweet": "text"})
        print(f"[shawky] Loaded 'real.csv': {len(df_real)} rows")
    except Exception as e:
        print(f"[shawky] WARNING: Could not load 'real.csv' - {e}")
        df_real = pd.DataFrame(columns=["text"])
    
    try:
        df_fake = load_df(SHAWKY_DIR, "fake.csv")
        df_fake = df_fake.rename(columns={"tweet": "text"})
        print(f"[shawky] Loaded 'fake.csv': {len(df_fake)} rows")
    except Exception as e:
        print(f"[shawky] WARNING: Could not load 'fake.csv' - {e}")
        df_fake = pd.DataFrame(columns=["text"])
    
    df_real['label'] = 1
    df_fake['label'] = 0
    
    df = pd.concat([df_real, df_fake], ignore_index=True)
    
    df['title'] = ""
    df['category'] = "Sports"
    df['dataset'] = SHAWKY
    
    return df


# --- Main Loader ---

"""
Kagglehub handles downloads (if needed) 
Combines all three fake-news datasets into one DataFrame.

Returns:

pd.DataFrame with columns:
    title    - article/tweet title ('' for tweet-only rows)
    text     - main text content
    label    - 0 = fake, 1 = real
    category - topic category ('Unknown' where unavailable)
    dataset  - origin dataset name
"""
def load_datasets() -> pd.DataFrame:
    print("-" * 60)
    print("Fake News Dataset Ingestion")
    print("-" * 60)
    
    print("\nLoading bhavikjikadara ...")
    df_bhavik = load_bhavik()
    
    print("\nLoading mahdimashayekhi ...")
    df_mahdi = load_mahdi()
    
    print("\nLoading shawkyelgendy ...")
    df_shawky = load_shawky()
    
    # Combine
    combined = pd.concat([df_bhavik, df_mahdi, df_shawky], ignore_index=True)
    combined = combined[["title", "text", "label", "category", "dataset"]]
    
    # Basic Preprocessing
    combined = preprocess_normalise_category(combined)
    combined = preprocess_drop_na_text(combined)
    combined = preprocess_drop_duplicates(combined)
    
    # Summary
    summarize_datasets(combined)
    
    return combined


# --- Data Persistence ---

def save_combined(df: pd.DataFrame, filename: str = "merged_fake_news.csv"):
    if DRIVE_ROOT.exists():
        data_dir = DRIVE_ROOT / "data"
    else:
        data_dir = Path("/content")
    
    # Save the combined DataFrame to CSV inside the data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / filename, index=False)
    
    print(f"Saved -> {data_dir / filename}")

