"""Metrics, uncertainty, error analysis and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Reference categorical palette (validated slots 1-3) + neutral for baselines.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"


def score_metrics(y, p, threshold: float) -> dict:
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision_neg": float(precision_score(y, yhat, zero_division=0)),
        "recall_neg": float(recall_score(y, yhat, zero_division=0)),
        "f1_neg": float(f1_score(y, yhat, zero_division=0)),
        "flag_rate": float(yhat.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(len(y)), "prevalence": float(np.mean(y)),
    }


def hotel_bootstrap_ci(df: pd.DataFrame, p: np.ndarray, threshold: float,
                       n_boot: int = 300, seed: int = 42) -> dict:
    """95% CI by resampling HOTELS (the split unit), not rows.

    Reviews of the same hotel are correlated; a row bootstrap would give
    intervals that are too narrow.
    """
    rng = np.random.default_rng(seed)
    y = df["is_negative"].to_numpy()
    idx_by_hotel = df.groupby("Hotel_Name").indices
    hotels = np.array(list(idx_by_hotel))
    pr, rec, prec = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(hotels, size=len(hotels), replace=True)
        idx = np.concatenate([idx_by_hotel[h] for h in pick])
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        pr.append(average_precision_score(yb, pb))
        yhat = pb >= threshold
        rec.append(recall_score(yb, yhat, zero_division=0))
        prec.append(precision_score(yb, yhat, zero_division=0))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"pr_auc_95ci": q(pr), "recall_neg_95ci": q(rec),
            "precision_neg_95ci": q(prec), "n_boot": len(pr)}


def top_terms(pipeline, k: int = 25) -> pd.DataFrame:
    vocab = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    coef = pipeline.named_steps["clf"].coef_.ravel()
    order = np.argsort(coef)
    neg = pd.DataFrame({"term": vocab[order[::-1][:k]], "coef": coef[order[::-1][:k]],
                        "direction": "-> negative"})
    pos = pd.DataFrame({"term": vocab[order[:k]], "coef": coef[order[:k]],
                        "direction": "-> positive"})
    return pd.concat([neg, pos], ignore_index=True)


def error_analysis(test: pd.DataFrame, p: np.ndarray, threshold: float, k: int = 25) -> pd.DataFrame:
    t = test.assign(p_negative=p, predicted=(p >= threshold).astype(int))
    fn = t[(t.is_negative == 1) & (t.predicted == 0)].nsmallest(k, "p_negative").assign(error="false_negative")
    fp = t[(t.is_negative == 0) & (t.predicted == 1)].nlargest(k, "p_negative").assign(error="false_positive")
    cols = ["error", "Hotel_Name", "Reviewer_Score", "p_negative", "text"]
    return pd.concat([fn, fp])[cols]


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)


def plot_pr_curves(y, curves: dict[str, np.ndarray], out: Path):
    colors = [BLUE, ORANGE, AQUA, GREY]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    for (name, p), c in zip(curves.items(), colors):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=c, linewidth=2,
                label=f"{name}  (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(np.mean(y), color=GREY, linestyle="--", linewidth=1,
               label=f"prior / no-skill ({np.mean(y):.3f})")
    ax.set_xlabel("Recall (share of negative reviews caught)", color=MUTED)
    ax.set_ylabel("Precision (share of flags that are negative)", color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax, "Precision-recall on held-out hotels")
    ax.legend(frameon=False, fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.42))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(m: dict, out: Path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
    ax.imshow(cm, cmap="Blues")
    labels = ["positive", "negative"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Actual", color=MUTED)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=11)
    ax.set_title(f"Confusion matrix at threshold {m['threshold']:.2f}", loc="left", fontsize=11, color=INK)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_calibration(y, p, out: Path):
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot(mean_pred, frac, color=BLUE, linewidth=2, marker="o", markersize=5, label="model")
    ax.set_xlabel("Predicted probability of negative", color=MUTED)
    ax.set_ylabel("Observed share negative", color=MUTED)
    _style(ax, "Calibration (held-out hotels)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
