"""Schema-identical synthetic data so tests and CI run without the 230 MB Kaggle file.

It checks that the pipeline RUNS correctly; it says nothing about real performance.
"""

import numpy as np
import pandas as pd

POS = ["great location", "friendly staff", "lovely breakfast", "very clean room",
       "comfortable bed", "excellent value", "helpful reception", "quiet room"]
NEG = ["room was not clean", "staff were rude", "no hot water", "very noisy at night",
       "breakfast was cold", "bed was uncomfortable", "wifi did not work", "overpriced"]


def make_reviews(n: int = 1500, n_hotels: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        unhappy = rng.random() < 0.25
        score = float(rng.choice([2.5, 3.8, 4.2, 5.0, 6.3])) if unhappy else float(rng.choice([7.5, 8.3, 9.2, 10.0]))
        k_neg = rng.integers(2, 4) if unhappy else rng.integers(0, 2)
        k_pos = rng.integers(0, 2) if unhappy else rng.integers(1, 4)
        neg = ". ".join(rng.choice(NEG, size=k_neg)) if k_neg else "No Negative"
        pos = ". ".join(rng.choice(POS, size=k_pos)) if k_pos else "No Positive"
        month, day, year = rng.integers(1, 13), rng.integers(1, 28), rng.choice([2015, 2016, 2017])
        rows.append({
            "Hotel_Name": f"Hotel {rng.integers(n_hotels)}",
            "Review_Date": f"{month}/{day}/{year}",
            "Positive_Review": f" {pos} ",
            "Negative_Review": f" {neg} ",
            "Reviewer_Score": score,
        })
    return pd.DataFrame(rows)
