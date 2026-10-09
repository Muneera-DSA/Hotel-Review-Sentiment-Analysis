"""Loading, labelling, de-duplication and leakage-safe splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .preprocess import normalise, strip_placeholder

REQUIRED_COLUMNS = [
    "Hotel_Name",
    "Review_Date",
    "Positive_Review",
    "Negative_Review",
    "Reviewer_Score",
]
# Carried through for subgroup analysis only; never used as model inputs.
CONTEXT_COLUMNS = ["Reviewer_Nationality", "Hotel_Address"]


def load_raw(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'Hotel_Reviews.csv' from "
            "https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe "
            "and place it in the data/ folder (see data/README.md)."
        )
    header = pd.read_csv(path, nrows=0).columns
    missing = set(REQUIRED_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    cols = REQUIRED_COLUMNS + [c for c in CONTEXT_COLUMNS if c in header]
    df = pd.read_csv(path, usecols=cols)
    validate_raw(df)
    return df


def validate_raw(df: pd.DataFrame) -> None:
    """Fail fast if the file is not the dataset the pipeline was built for."""
    problems = []
    if df["Reviewer_Score"].isna().any():
        problems.append("missing Reviewer_Score values")
    if not df["Reviewer_Score"].between(1, 10).all():
        problems.append("Reviewer_Score outside 1-10")
    if df["Hotel_Name"].isna().any():
        problems.append("missing Hotel_Name values")
    dates = pd.to_datetime(df["Review_Date"], format="%m/%d/%Y", errors="coerce")
    if dates.isna().mean() > 0.01:
        problems.append("more than 1% of Review_Date values unparseable (expected m/d/YYYY)")
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))


def label_from_score(score: pd.Series,
                     neg_max: float = config.NEGATIVE_MAX_SCORE,
                     pos_min: float = config.POSITIVE_MIN_SCORE) -> pd.Series:
    """1 = negative review, 0 = positive review, NaN = ambiguous middle band (dropped)."""
    if neg_max >= pos_min:
        raise ValueError("neg_max must be below pos_min")
    label = pd.Series(np.nan, index=score.index)
    label[score <= neg_max] = 1
    label[score >= pos_min] = 0
    return label


def build_dataset(raw: pd.DataFrame,
                  neg_max: float = config.NEGATIVE_MAX_SCORE,
                  pos_min: float = config.POSITIVE_MIN_SCORE) -> tuple[pd.DataFrame, dict]:
    """Turn raw rows into one labelled document per review.

    Returns the modelling frame and an audit dict of rows removed at each step,
    which is written to reports/ so every exclusion is traceable.
    """
    audit = {"raw_rows": len(raw)}
    df = raw.drop_duplicates(subset=REQUIRED_COLUMNS).copy()
    audit["after_exact_duplicate_removal"] = len(df)

    pos = df["Positive_Review"].map(strip_placeholder)
    neg = df["Negative_Review"].map(strip_placeholder)
    # One document per review, as a guest would write it in free text.
    # The field names are deliberately NOT encoded: in production there are no
    # separate "liked" / "disliked" boxes.
    df["text"] = (pos + " " + neg).map(normalise)
    df = df[df["text"].str.len() > 0]
    audit["after_empty_text_removal"] = len(df)

    df["is_negative"] = label_from_score(df["Reviewer_Score"], neg_max, pos_min)
    audit["ambiguous_band_removed"] = int(df["is_negative"].isna().sum())
    df = df.dropna(subset=["is_negative"])
    df["is_negative"] = df["is_negative"].astype(int)

    # Same text + same hotel posted twice is a duplicate review, keep one.
    df = df.drop_duplicates(subset=["Hotel_Name", "text"])
    audit["after_text_duplicate_removal"] = len(df)

    df["Review_Date"] = pd.to_datetime(df["Review_Date"], format="%m/%d/%Y", errors="coerce")
    audit["final_rows"] = len(df)
    audit["negative_rate"] = float(df["is_negative"].mean()) if len(df) else float("nan")
    audit["n_hotels"] = int(df["Hotel_Name"].nunique())
    keep = ["Hotel_Name", "Review_Date", "text", "Reviewer_Score", "is_negative"]
    keep += [c for c in CONTEXT_COLUMNS if c in df.columns]
    return df.reset_index(drop=True)[keep], audit


def split_by_hotel(df: pd.DataFrame,
                   test_size: float = config.TEST_SIZE,
                   random_state: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary split: test hotels are never seen in training.

    A random row split lets the model memorise hotel-specific vocabulary
    (hotel names, street names, recurring complaints about one property)
    and overstates performance on new properties.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr, te = next(gss.split(df, df["is_negative"], groups=df["Hotel_Name"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def split_by_time(df: pd.DataFrame, test_size: float = config.TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robustness split: train on older reviews, test on the most recent ones (drift check)."""
    d = df.dropna(subset=["Review_Date"]).sort_values("Review_Date")
    cutoff = d["Review_Date"].quantile(1 - test_size)
    return (d[d["Review_Date"] < cutoff].reset_index(drop=True),
            d[d["Review_Date"] >= cutoff].reset_index(drop=True))
