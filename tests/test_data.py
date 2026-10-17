import numpy as np
import pandas as pd
import pytest

from hotel_sentiment.data import build_dataset, label_from_score, split_by_hotel, split_by_time
from synthetic import make_reviews


def test_label_bands():
    s = pd.Series([2.5, 5.0, 5.1, 7.9, 8.0, 10.0])
    lab = label_from_score(s, neg_max=5.0, pos_min=8.0)
    assert lab.tolist()[:2] == [1, 1]
    assert np.isnan(lab.iloc[2]) and np.isnan(lab.iloc[3])
    assert lab.tolist()[4:] == [0, 0]


def test_label_bands_validated():
    with pytest.raises(ValueError):
        label_from_score(pd.Series([5.0]), neg_max=8.0, pos_min=5.0)


def test_build_dataset_removes_placeholders_duplicates_and_ambiguous():
    raw = make_reviews(800)
    raw = pd.concat([raw, raw.head(50)], ignore_index=True)  # inject exact duplicates
    df, audit = build_dataset(raw)
    assert audit["after_exact_duplicate_removal"] <= audit["raw_rows"] - 50
    assert not df["text"].str.contains("no negative|no positive").any()
    assert df["Reviewer_Score"].between(5.0, 8.0, inclusive="neither").sum() == 0
    assert not df.duplicated(["Hotel_Name", "text"]).any()
    assert set(df["is_negative"].unique()) <= {0, 1}


def test_no_hotel_in_both_train_and_test():
    df, _ = build_dataset(make_reviews(1500, n_hotels=40))
    train, test = split_by_hotel(df)
    assert set(train.Hotel_Name).isdisjoint(test.Hotel_Name)
    assert len(train) + len(test) == len(df)


def test_time_split_is_chronological():
    df, _ = build_dataset(make_reviews(1500))
    train, test = split_by_time(df)
    assert train["Review_Date"].max() < test["Review_Date"].min()


def test_validate_raw_rejects_bad_scores():
    from hotel_sentiment.data import validate_raw
    raw = make_reviews(50)
    raw.loc[0, "Reviewer_Score"] = 42
    with pytest.raises(ValueError):
        validate_raw(raw)


def test_label_thresholds_are_configurable():
    raw = make_reviews(800)
    strict, _ = build_dataset(raw, neg_max=4.0, pos_min=9.0)
    loose, _ = build_dataset(raw, neg_max=6.0, pos_min=7.0)
    assert len(strict) < len(loose)
