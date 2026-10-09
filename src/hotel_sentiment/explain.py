"""SHAP explanations for the TF-IDF + logistic regression pipeline.

    python -m hotel_sentiment.explain        # after python -m hotel_sentiment.train

For a linear model with (assumed) independent features, SHAP values have an exact
closed form (Lundberg & Lee, 2017; the "interventional" LinearExplainer in the shap
package):

    phi_j(x) = w_j * (x_j - E[x_j])        base value = w . E[x] + b

so phi summed over all features + base value = the model's log-odds for x, exactly.
E[x] is the mean TF-IDF vector of ALL training reviews (the background set).
Note: shap.LinearExplainer's default masker subsamples the background to 100
rows, so its values differ slightly unless the full background is passed.
We compute this directly on the sparse matrix; tests/test_explain.py checks it
against shap.LinearExplainer when the shap package is installed.

Values are in log-odds of "negative review". Positive phi pushes towards negative.

How this differs from reports/top_terms.csv (raw coefficients): a coefficient says
how strongly a word moves the score *when it appears*; SHAP also accounts for how
often and how strongly it appears, so a rare extreme word ranks lower globally than
a common, moderately strong one.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from . import config  # noqa: E402
from .data import build_dataset, load_raw, split_by_hotel  # noqa: E402
from .evaluate import BLUE, INK, MUTED, ORANGE  # noqa: E402
from .predict import load_artifact  # noqa: E402


class LinearShap:
    """Exact SHAP values for pipeline = TfidfVectorizer -> LogisticRegression."""

    def __init__(self, pipeline, background_texts):
        self.vec = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        self.w = clf.coef_.ravel()
        self.b = float(clf.intercept_[0])
        self.vocab = np.asarray(self.vec.get_feature_names_out())
        self.mean_x = np.asarray(self.vec.transform(background_texts).mean(axis=0)).ravel()
        self.base_value = float(self.w @ self.mean_x + self.b)
        # Contribution every feature makes when ABSENT from a review (x_j = 0).
        self._absent = -self.w * self.mean_x
        self._absent_total = float(self._absent.sum())

    def transform(self, texts) -> sp.csr_matrix:
        return self.vec.transform(texts)

    def shap_dense(self, X: sp.csr_matrix) -> np.ndarray:
        """Full SHAP matrix (n x n_features). Use only for small n."""
        return (X.toarray() - self.mean_x) * self.w

    def explain_one(self, X_row: sp.csr_matrix, top: int = 10) -> dict:
        """Top present features + the summed contribution of everything else."""
        idx = X_row.indices
        phi_present = self.w[idx] * (X_row.data - self.mean_x[idx])
        rest_absent = self._absent_total - float(self._absent[idx].sum())
        order = np.argsort(-np.abs(phi_present))
        shown, hidden = order[:top], order[top:]
        logit = self.base_value + float(phi_present.sum()) + rest_absent
        return {
            "base_value": self.base_value,
            "features": [(str(self.vocab[idx[i]]), float(phi_present[i])) for i in shown],
            "other_present_features": float(phi_present[hidden].sum()),
            "absent_features": rest_absent,
            "logit": logit,
            "p_negative": float(1 / (1 + np.exp(-logit))),
        }

    def global_importance(self, X: sp.csr_matrix, top: int = 20) -> pd.DataFrame:
        """Mean |SHAP| per feature over X, computed sparsely.

        For feature j: rows where x_j > 0 contribute |w_j (x_ij - m_j)|; the
        remaining rows each contribute |w_j m_j|.
        """
        n = X.shape[0]
        Xc = X.tocsc()
        present_abs = np.zeros(X.shape[1])
        n_present = np.diff(Xc.indptr)
        for j in np.nonzero(n_present)[0]:
            vals = Xc.data[Xc.indptr[j]:Xc.indptr[j + 1]]
            present_abs[j] = np.abs(self.w[j] * (vals - self.mean_x[j])).sum()
        mean_abs = (present_abs + (n - n_present) * np.abs(self.w * self.mean_x)) / n
        order = np.argsort(-mean_abs)[:top]
        return pd.DataFrame({
            "term": self.vocab[order],
            "mean_abs_shap": mean_abs[order],
            "coefficient": self.w[order],
            "share_of_reviews_containing": n_present[order] / n,
            "direction": np.where(self.w[order] > 0, "-> negative", "-> positive"),
        })


# ---------------------------------------------------------------------------- plots

def _clean_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_global(imp: pd.DataFrame, out: Path):
    d = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=150)
    colors = [ORANGE if c > 0 else BLUE for c in d["coefficient"]]
    ax.barh(d["term"], d["mean_abs_shap"], color=colors, height=0.7)
    ax.set_xlabel("Mean |SHAP value| (log-odds), held-out hotels", color=MUTED)
    ax.set_title("Which words drive predictions overall", loc="left", fontsize=11, color=INK)
    ax.bar(0, 0, color=ORANGE, label="pushes towards negative")
    ax.bar(0, 0, color=BLUE, label="pushes towards positive")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(left=0)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_waterfalls(examples: list[dict], out: Path):
    fig, axes = plt.subplots(len(examples), 1, figsize=(7.2, 3.3 * len(examples)), dpi=150)
    for ax, ex in zip(np.atleast_1d(axes), examples):
        e = ex["explanation"]
        labels = [t for t, _ in e["features"]] + ["other words in review", "words not in review"]
        vals = [v for _, v in e["features"]] + [e["other_present_features"], e["absent_features"]]
        start = e["base_value"]
        lefts = []
        for v in vals:
            lefts.append(start if v >= 0 else start + v)
            start += v
        y = np.arange(len(vals))[::-1]
        ax.barh(y, np.abs(vals), left=lefts, color=[ORANGE if v > 0 else BLUE for v in vals], height=0.65)
        ax.set_yticks(y, labels)
        ax.axvline(e["base_value"], color="#8a8984", linestyle="--", linewidth=1)
        ax.axvline(e["logit"], color=INK, linewidth=1)
        title = (f"{ex['title']}  ·  guest score {ex['score']}  ·  model P(negative) = {e['p_negative']:.2f}\n"
                 + textwrap.shorten(ex["text"], 95, placeholder=" …"))
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("log-odds of negative  (dashed = average review, solid = this review)", color=MUTED, fontsize=8)
        _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def _pick_examples(test: pd.DataFrame, p: np.ndarray, threshold: float) -> list[tuple[str, int]]:
    """Deterministic, representative cases (not cherry-picked by hand)."""
    t = test.assign(p=p, words=test["text"].str.split().str.len()).reset_index(drop=True)
    mid = t[(t.words.between(15, 45))]
    picks = []
    tp = mid[(mid.is_negative == 1) & (mid.p >= threshold)]
    picks.append(("Correctly flagged negative", int((tp.p - tp.p.median()).abs().idxmin())))
    mixed = mid[(mid.is_negative == 0) & (mid.p.between(0.3, 0.6))]
    picks.append(("Mixed review, positive score, borderline", int((mixed.p - 0.45).abs().idxmin())))
    fn = mid[(mid.is_negative == 1) & (mid.p < 0.05)]
    picks.append(("Missed: low score, positive text (label noise?)", int(fn.p.idxmin())))
    return picks, t


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.DATA_PATH)
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    args = ap.parse_args(argv)

    art = load_artifact(args.model_path)
    pipe, threshold = art["pipeline"], art["threshold"]
    df, _ = build_dataset(load_raw(args.data))
    train, test = split_by_hotel(df)        # identical split to training (fixed seed)

    explainer = LinearShap(pipe, train["text"])
    Xte = explainer.transform(test["text"])
    p = pipe.predict_proba(test["text"])[:, 1]

    # Sanity check: SHAP values must add up to the model output exactly.
    sample = Xte[:500]
    phi = explainer.shap_dense(sample)
    recon = explainer.base_value + phi.sum(axis=1)
    max_err = float(np.abs(recon - pipe.decision_function(test["text"].iloc[:500])).max())
    assert max_err < 1e-6, f"SHAP additivity violated: {max_err}"

    imp = explainer.global_importance(Xte)
    figs = args.reports_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    imp.to_csv(args.reports_dir / "shap_global.csv", index=False)
    plot_global(imp, figs / "shap_global.png")

    picks, t = _pick_examples(test, p, threshold)
    examples = []
    for title, i in picks:
        e = explainer.explain_one(Xte[i])
        examples.append({"title": title, "score": float(t.loc[i, "Reviewer_Score"]), "hotel": t.loc[i, "Hotel_Name"],
                         "text": t.loc[i, "text"], "explanation": e})
    plot_waterfalls(examples, figs / "shap_examples.png")
    summary = {"base_value_logodds": explainer.base_value,
               "base_value_probability": float(1 / (1 + np.exp(-explainer.base_value))),
               "additivity_max_abs_error": max_err,
               "examples": examples}
    (args.reports_dir / "shap_examples.json").write_text(json.dumps(summary, indent=2))
    print(imp.to_string(index=False))
    print(f"additivity check max error: {max_err:.2e}")
    return summary


if __name__ == "__main__":
    main()
