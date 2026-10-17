"""A broad screen of model families on the training hotels (what LazyPredict does, done properly).

    python experiments/model_screen.py

Writes reports/model_screen.{json,md}.

- Uses only the **training hotels** of the main split; the test hotels stay untouched.
- 3-fold GroupKFold by hotel, so every fold is scored on hotels its model never saw.
- The TF-IDF vocabulary is fitted inside each fold, on that fold's training hotels only.
- Tree ensembles can't use 100,000 sparse columns efficiently, so they get a 200-dimensional
  TruncatedSVD (latent semantic analysis) of the same TF-IDF, also fitted inside the fold.
- Each model is compared with the best one using the Nadeau-Bengio corrected resampled t-test,
  with Holm adjustment. With only three folds the test has little power; large differences are
  still clear.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hotel_sentiment import config  # noqa: E402
from hotel_sentiment.data import build_dataset, load_raw, split_by_hotel  # noqa: E402

SEED = config.RANDOM_STATE
C = float(json.loads((config.REPORTS_DIR / "metrics.json").read_text())["model"]["best_C"])

SPARSE = {
    "Prior only (no information)": DummyClassifier(strategy="prior"),
    "Logistic regression (final model)": LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=SEED),
    "Multinomial naive Bayes": MultinomialNB(alpha=0.1),
    "Complement naive Bayes": ComplementNB(alpha=0.3),
    "Bernoulli naive Bayes": BernoulliNB(alpha=0.1, binarize=0.0),
    "Linear SVM": LinearSVC(C=0.1, random_state=SEED),
    "SGD (modified Huber loss)": SGDClassifier(loss="modified_huber", alpha=1e-5, random_state=SEED),
    "Ridge classifier": RidgeClassifier(alpha=3.0),
    "Passive-aggressive": PassiveAggressiveClassifier(C=0.01, random_state=SEED),
}
DENSE = {   # on a 200-dimensional TruncatedSVD of the same TF-IDF
    "LSA-200 + logistic regression": LogisticRegression(max_iter=2000),
    "LSA-200 + histogram gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED),
    "LSA-200 + random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
}


def _score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def corrected_t(diffs: np.ndarray, n_train: int, n_test: int) -> float:
    k, var = len(diffs), np.var(diffs, ddof=1)
    if var == 0:
        return 1.0
    t = diffs.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def main() -> dict:
    df, _ = build_dataset(load_raw())
    train, _test = split_by_hotel(df)
    y = train["is_negative"].to_numpy()
    folds = list(GroupKFold(n_splits=3).split(train, y, groups=train["Hotel_Name"]))
    scores = {name: {"auc": [], "pr_auc": [], "seconds": 0.0} for name in list(SPARSE) + list(DENSE)}
    for f, (tr, va) in enumerate(folds):
        t0 = time.time()
        vec = TfidfVectorizer(**config.TFIDF_PARAMS).fit(train["text"].iloc[tr])
        Xtr, Xva = vec.transform(train["text"].iloc[tr]), vec.transform(train["text"].iloc[va])
        svd = TruncatedSVD(200, random_state=SEED).fit(Xtr)
        Ztr, Zva = svd.transform(Xtr), svd.transform(Xva)
        print(f"fold {f + 1}: features ready in {time.time() - t0:.0f} s", flush=True)
        for group, (A, B) in (("sparse", (Xtr, Xva)), ("dense", (Ztr, Zva))):
            for name, est in (SPARSE if group == "sparse" else DENSE).items():
                t1 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = clone(est).fit(A, y[tr])
                s = _score(m, B)
                scores[name]["auc"].append(float(roc_auc_score(y[va], s)))
                scores[name]["pr_auc"].append(float(average_precision_score(y[va], s)))
                scores[name]["seconds"] += time.time() - t1
                print(f"   {name}: PR-AUC {scores[name]['pr_auc'][-1]:.3f}", flush=True)
    rows = [{"model": n, "pr_auc": float(np.mean(v["pr_auc"])), "pr_auc_sd": float(np.std(v["pr_auc"], ddof=1)),
             "roc_auc": float(np.mean(v["auc"])), "fold_pr_auc": v["pr_auc"], "seconds": round(v["seconds"], 1)}
            for n, v in scores.items()]
    rows.sort(key=lambda r: -r["pr_auc"])
    best = rows[0]["model"]
    n_va = len(y) // 3
    best_folds = np.array(scores[best]["pr_auc"])
    raw = {r["model"]: corrected_t(best_folds - np.array(r["fold_pr_auc"]), len(y) - n_va, n_va)
           for r in rows if r["model"] != best}
    adj = holm(raw)
    for r in rows:
        r["p_vs_best"], r["p_vs_best_holm"] = raw.get(r["model"], 1.0), adj.get(r["model"], 1.0)
    out = {"n_training_reviews": int(len(y)), "n_training_hotels": int(train.Hotel_Name.nunique()),
           "folds": "3-fold GroupKFold by hotel (training hotels only)",
           "primary_metric": f"PR-AUC (negative reviews are {y.mean():.1%} of training reviews)", "best": best, "table": rows}
    (config.REPORTS_DIR / "model_screen.json").write_text(json.dumps(out, indent=1))
    L = ["# Model screen (generated by `experiments/model_screen.py`)", "",
         f"{out['n_training_reviews']:,} reviews from {out['n_training_hotels']:,} training hotels; {out['folds']}. "
         "TF-IDF (and the SVD for tree models) fitted inside each fold. "
         "p-values: corrected resampled t-test against the best model, Holm-adjusted (three folds, so low power).", "",
         "| Model | PR-AUC (mean ± SD) | ROC-AUC | p vs best (Holm) | Fit time, s |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['pr_auc']:.3f} ± {r['pr_auc_sd']:.3f} | {r['roc_auc']:.3f} "
                 f"| {r['p_vs_best_holm']:.3f} | {r['seconds']:.0f} |")
    (config.REPORTS_DIR / "model_screen.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return out


if __name__ == "__main__":
    main()
