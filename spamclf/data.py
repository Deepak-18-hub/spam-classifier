"""Load the UCI SMS Spam Collection. One loader, shared by the notebook and training."""
import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "sms_spam" / "SMSSpamCollection"
EXPECTED_ROWS = 5574  # from the dataset readme


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return a DataFrame with columns label ('ham'/'spam') and text.

    The file is tab-separated with no header. QUOTE_NONE matters: 145 messages contain a
    double quote, and with default CSV quoting pandas silently drops rows instead of failing.
    """
    df = pd.read_csv(
        path, sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(df)}")
    return df


def load_split(seed: int = SEED):
    """Drop exact duplicates, then a stratified 80/20 split.

    Every model must use this one function so they are all scored on the identical test set.
    Returns X_train, X_test, y_train, y_test (y: 1 = spam) and the number of unique messages.
    """
    df = load_raw().drop_duplicates().reset_index(drop=True)
    X = df["text"].tolist()
    y = (df["label"] == "spam").astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    return X_train, X_test, y_train, y_test, len(df)
