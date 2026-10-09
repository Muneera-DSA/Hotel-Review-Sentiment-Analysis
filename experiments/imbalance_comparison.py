"""Does resampling (SMOTE, oversampling) beat class weighting? And what does
SMOTE-before-split do to the reported numbers?

    python experiments/imbalance_comparison.py

Writes reports/imbalance_comparison.{json,md}. Uses the same cleaned data,
hotel split and TF-IDF settings as the main pipeline, with C read from the
saved model (the value the main run selected) so only the balancing method varies.

SMOTE is implemented here (k=5 minority neighbours, uniform interpolation,
as in Chawla et al., 2002) rather than imported, so the exact procedure is
visible and works directly on sparse TF-IDF vectors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hotel_sentiment import config  # noqa: E402
from hotel_sentiment.data import build_dataset, load_raw, split_by_hotel  # noqa: E402

C = float(joblib.load(config.MODEL_PATH)["pipeline"].named_steps["clf"].C)
SEED = config.RANDOM_STATE


def _cosine_knn(X: sp.csr_matrix, k: int, chunk: int = 500) -> np.ndarray:
    """k nearest neighbours by cosine similarity, excluding self, computed in chunks.

    TF-IDF rows are L2-normalised, so cosine similarity is a dot product. Chunking keeps
    memory at chunk x n_minority instead of n_minority^2.
    """
    out = np.empty((X.shape[0], k), dtype=np.int64)
    XT = X.T.tocsc()
    for start in range(0, X.shape[0], chunk):
        sims = (X[start:start + chunk] @ XT).toarray()
        rows = np.arange(sims.shape[0])
        sims[rows, start + rows] = -np.inf          # exclude self
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        out[start:start + sims.shape[0]] = top
    return out


def smote(X: sp.csr_matrix, y: np.ndarray, k: int = 5, seed: int = SEED):
    """Oversample the minority class (1) to parity with synthetic interpolations."""
    rng = np.random.default_rng(seed)
    Xmin = X[y == 1]
    n_new = int((y == 0).sum() - (y == 1).sum())
    neigh = _cosine_knn(Xmin, k)
    base = rng.integers(0, Xmin.shape[0], n_new)
    partner = neigh[base, rng.integers(0, k, n_new)]
    gap = sp.diags(rng.random(n_new))
    synthetic = Xmin[base] + gap @ (Xmin[partner] - Xmin[base])
    return sp.vstack([X, synthetic]).tocsr(), np.concatenate([y, np.ones(n_new, dtype=int)])


def random_oversample(X, y, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx_min = np.where(y == 1)[0]
    extra = rng.choice(idx_min, int((y == 0).sum() - len(idx_min)), replace=True)
    return sp.vstack([X, X[extra]]).tocsr(), np.concatenate([y, y[extra]])


def lr(class_weight=None):
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, solver="liblinear", random_state=SEED)


def evaluate(y, p) -> dict:
    prec, rec, _ = precision_recall_curve(y, p)
    yhat = p >= 0.5
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        # Read off the test PR curve: a like-for-like ranking comparison across methods,
        # not a deployable operating point (that needs a threshold chosen on training data).
        "precision_at_80_recall": float(prec[:-1][rec[:-1] >= 0.80].max()),
        "recall_at_0.5": float(recall_score(y, yhat)),
        "precision_at_0.5": float(precision_score(y, yhat, zero_division=0)),
        "n_test": int(len(y)),
        "test_negative_rate": float(np.mean(y)),
    }


def main():
    df, _ = build_dataset(load_raw())
    results = {}

    # ---- Correct protocol: fit everything on training hotels only ----------------
    train, test = split_by_hotel(df)
    vec = TfidfVectorizer(**config.TFIDF_PARAMS)
    Xtr, Xte = vec.fit_transform(train["text"]), vec.transform(test["text"])
    ytr, yte = train["is_negative"].to_numpy(), test["is_negative"].to_numpy()

    runs = {
        "no balancing (current model)": (Xtr, ytr, None),
        "class_weight='balanced'": (Xtr, ytr, "balanced"),
        "random oversampling (train only)": (*random_oversample(Xtr, ytr), None),
        "SMOTE (train only)": (*smote(Xtr, ytr), None),
    }
    for name, (X, y, cw) in runs.items():
        p = lr(cw).fit(X, y).predict_proba(Xte)[:, 1]
        results[name] = {**evaluate(yte, p), "train_rows_after_resampling": int(X.shape[0])}
        print(name, {k: round(v, 3) for k, v in results[name].items() if isinstance(v, float)})

    del runs, Xtr, Xte, vec

    # ---- Leaky protocol: SMOTE on the full dataset, THEN split -------------------
    # Synthetic test rows are interpolations of training rows (and vice versa), so the
    # test set is no longer unseen data. Compared with the same random split, no SMOTE.
    vec_all = TfidfVectorizer(**config.TFIDF_PARAMS)
    X_all, y_all = vec_all.fit_transform(df["text"]), df["is_negative"].to_numpy()
    Xs, ys = smote(X_all, y_all)
    a, b, ya, yb = train_test_split(Xs, ys, test_size=config.TEST_SIZE, random_state=SEED, stratify=ys)
    del Xs, ys
    p = lr().fit(a, ya).predict_proba(b)[:, 1]
    results["LEAKY: SMOTE before split (random split)"] = evaluate(yb, p)

    a, b, ya, yb = train_test_split(X_all, y_all, test_size=config.TEST_SIZE, random_state=SEED, stratify=y_all)
    p = lr().fit(a, ya).predict_proba(b)[:, 1]
    results["reference: same random split, no SMOTE"] = evaluate(yb, p)

    out = config.REPORTS_DIR
    (out / "imbalance_comparison.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Imbalance handling comparison (generated by `experiments/imbalance_comparison.py`)",
        "",
        f"Rows 1-4: identical hotel split, TF-IDF fitted on training hotels only, C = {C}; only the balancing method varies.",
        "Rows 5-6: random row split, to show what applying SMOTE *before* splitting does to the reported numbers.",
        "",
        "| Method | Test negative rate | PR-AUC | ROC-AUC | Precision at 80% recall | Recall @ 0.5 | Precision @ 0.5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['test_negative_rate']:.3f} | {r['pr_auc']:.3f} | {r['roc_auc']:.3f} | "
                     f"{r['precision_at_80_recall']:.3f} | {r['recall_at_0.5']:.3f} | {r['precision_at_0.5']:.3f} |")
    (out / "imbalance_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
