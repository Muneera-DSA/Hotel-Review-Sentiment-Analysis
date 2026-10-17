"""Central configuration: paths, labelling thresholds and model settings.

Everything a reviewer might want to question lives here, in one place.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "Hotel_Reviews.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Target definition -------------------------------------------------------
# The dataset has no sentiment label. We derive one from Reviewer_Score (2.5-10),
# which the guest gave alongside the text. The middle band is ambiguous and is
# excluded rather than forced into a class. The positive class (1) is the
# business-relevant one: an unhappy guest whose review needs a response.
NEGATIVE_MAX_SCORE = 5.0   # score <= 5.0  -> is_negative = 1
POSITIVE_MIN_SCORE = 8.0   # score >= 8.0  -> is_negative = 0
                           # 5.0 < score < 8.0 -> dropped (ambiguous)

# Booking.com placeholders inserted when a guest left a field empty.
# They are dataset artefacts, not guest language, and correlate strongly with
# the score, so they are removed before modelling.
PLACEHOLDERS = ("No Negative", "No Positive")

# --- Model ---------------------------------------------------------------------
TFIDF_PARAMS = {
    "ngram_range": (1, 2),      # bigrams keep "not clean", "no hot", "very good"
    "min_df": 5,                # ignore very rare terms (noise, typos)
    "max_df": 0.9,
    "max_features": 100_000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}
# No class re-weighting. Validated in reports/validation.md (section 3) and
# reports/imbalance_comparison.md: class_weight="balanced", oversampling and SMOTE
# do not improve ranking (balanced is slightly worse, PR-AUC -0.004) and they distort
# predicted probabilities. Imbalance is handled by choosing the threshold explicitly.
CLASS_WEIGHT = None
C_GRID = [0.25, 1.0, 4.0]
CV_FOLDS = 5

# Threshold policy: choose the probability cut-off on out-of-fold training
# predictions (never on the test set) to reach this recall on negative reviews.
TARGET_RECALL = 0.80
