import json

import numpy as np

from hotel_sentiment import predict as predict_mod
from hotel_sentiment.model import build_pipeline, threshold_for_recall
from hotel_sentiment.train import main
from synthetic import make_reviews


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    scores = np.clip(y * 0.4 + rng.random(2000) * 0.6, 0, 1)
    t = threshold_for_recall(y, scores, target_recall=0.8)
    assert ((scores >= t) & (y == 1)).sum() / y.sum() >= 0.8


def test_pipeline_learns_negation_bigram():
    texts = ["room was clean"] * 50 + ["room was not clean"] * 50
    y = [0] * 50 + [1] * 50
    pipe = build_pipeline(min_df=1).fit(texts, y)
    p = pipe.predict_proba(["the room was not clean", "the room was clean"])[:, 1]
    assert p[0] > p[1]


def test_end_to_end_train_and_predict(tmp_path):
    csv = tmp_path / "Hotel_Reviews.csv"
    make_reviews(3000, n_hotels=60, seed=1).to_csv(csv, index=False)
    model_path = tmp_path / "model.joblib"
    metrics = main(["--data", str(csv), "--reports-dir", str(tmp_path / "reports"),
                    "--model-path", str(model_path), "--n-boot", "20", "--n-jobs", "1"])

    reports = tmp_path / "reports"
    for f in ["metrics.json", "metrics.md", "top_terms.csv", "error_analysis.csv",
              "figures/pr_curve.png", "figures/confusion_matrix.png", "figures/calibration.png"]:
        assert (reports / f).exists(), f
    saved = json.loads((reports / "metrics.json").read_text())
    assert saved["split"]["train_hotels"] + saved["split"]["test_hotels"] == saved["data"]["n_hotels"]
    # The model must beat the no-skill prior on this (easy) synthetic data.
    assert metrics["test"]["model"]["pr_auc"] > metrics["test"]["baselines"]["prior"]["pr_auc"]

    predict_mod.load_artifact.cache_clear()
    out = predict_mod.predict(["Staff were rude and there was no hot water", "Lovely breakfast, friendly staff"],
                              model_path=model_path)
    assert out[0]["score_negative"] > out[1]["score_negative"]
    assert {r["label"] for r in out} <= {"negative", "positive"}
