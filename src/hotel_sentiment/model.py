"""Model, baselines and threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config


def build_pipeline(C: float = 1.0, **tfidf_overrides) -> Pipeline:
    """TF-IDF + logistic regression in ONE object.

    Saving a single Pipeline (rather than a vectorizer and a model separately)
    guarantees that inference applies exactly the transformation fitted in
    training, and makes it impossible to fit the vectorizer on test data.
    """
    tfidf = {**config.TFIDF_PARAMS, **tfidf_overrides}
    return Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf)),
        ("clf", LogisticRegression(
            C=C,
            class_weight=config.CLASS_WEIGHT,   # None: see config.py and reports/validation.md
            max_iter=2000,
            solver="liblinear",
            random_state=config.RANDOM_STATE,
        )),
    ])


def _word_count(texts):
    arr = np.asarray([len(t.split()) for t in texts], dtype=float)
    return np.log1p(arr).reshape(-1, 1)


def baselines() -> dict[str, Pipeline | DummyClassifier]:
    """Reference points the text model must beat.

    - prior: predicts the base rate. Its PR-AUC equals the negative-review rate.
    - review_length: unhappy guests tend to write more. If TF-IDF barely beats
      this, the text model is not learning much language.
    - tfidf_unigram: ablation, shows what bigrams add.
    - tfidf_stopwords_removed: ablation, sklearn's stopword list contains
      "not"/"no"/"nor"; shows the cost of removing negations (the original
      version of this project did this via NLTK).
    """
    stop = list(ENGLISH_STOP_WORDS)
    return {
        "prior": DummyClassifier(strategy="prior"),
        "review_length": Pipeline([
            ("len", FunctionTransformer(_word_count)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced")),
        ]),
        "tfidf_unigram": build_pipeline(ngram_range=(1, 1)),
        "tfidf_stopwords_removed": build_pipeline(stop_words=stop),
    }


def threshold_for_recall(y_true, scores, target_recall: float = config.TARGET_RECALL) -> float:
    """Highest threshold whose recall >= target (i.e. best precision at that recall).

    Must be called on out-of-fold TRAINING predictions; choosing a threshold
    on the test set would leak test information into the reported metrics.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds.min())
    return float(thresholds[ok[-1]])
