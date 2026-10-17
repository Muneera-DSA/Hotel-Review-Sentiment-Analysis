"""Validation suite: is the model's performance real, stable, and fit for its purpose?

    python -m hotel_sentiment.validate       # after python -m hotel_sentiment.train

Writes reports/validation.md and reports/validation.json. Sections:

  1. Data fingerprint            - SHA-256 of the input file, so results are tied to exact data
  2. Repeated hotel splits       - is the headline number stable across different test hotels?
  3. Paired significance tests   - are the design choices (bigrams, keeping negations,
                                   class weights) real improvements or noise?
  4. Label-definition sensitivity- do conclusions hold if "negative" is defined differently?
  5. Unseen-text subset          - performance on reviews whose exact text never appears in training
  6. Behavioural tests           - minimal-pair checks: negation, negated complaints, invariance
  7. Subgroups                   - review length, reviewer nationality, city
  8. Operating-point trade-off   - reviews to read vs unhappy guests caught, per 1,000 reviews

Every model here uses the regularisation strength selected by the main training run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config
from .data import build_dataset, load_raw, split_by_hotel
from .model import build_pipeline, threshold_for_recall
from .predict import load_artifact
from .preprocess import normalise

ENGLISH_NATIVE = {"United Kingdom", "United States of America", "Ireland", "Australia", "Canada", "New Zealand"}
COUNTRY_TO_CITY = {"Netherlands": "Amsterdam", "Spain": "Barcelona", "United Kingdom": "London",
                   "Italy": "Milan", "France": "Paris", "Austria": "Vienna"}


# ------------------------------------------------------------------ helpers

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prec_at_recall(y, p, target=0.80) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    return float(prec[:-1][rec[:-1] >= target].max())


def ranking_metrics(y, p) -> dict:
    y = np.asarray(y)
    if y.min() == y.max():
        return {"n": int(len(y)), "negative_rate": float(y.mean()), "pr_auc": None, "roc_auc": None}
    return {"n": int(len(y)), "negative_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "precision_at_80_recall": prec_at_recall(y, p)}


def at_threshold(y, p, t) -> dict:
    y, yhat = np.asarray(y), np.asarray(p) >= t
    tp = int((yhat & (y == 1)).sum())
    return {"recall": tp / max(int(y.sum()), 1), "precision": tp / max(int(yhat.sum()), 1),
            "flag_rate": float(yhat.mean())}


def fit_predict(C, train, test, **tfidf):
    m = build_pipeline(C=C, **tfidf).fit(train["text"], train["is_negative"])
    return m.predict_proba(test["text"])[:, 1]


def paired_bootstrap(test, p_a, p_b, n_boot=1000, seed=0) -> dict:
    """PR-AUC(a) - PR-AUC(b), resampling hotels (the split unit)."""
    rng = np.random.default_rng(seed)
    y = test["is_negative"].to_numpy()
    groups = test.groupby("Hotel_Name").indices
    hotels = np.array(list(groups))
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[h] for h in rng.choice(hotels, len(hotels), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(average_precision_score(y[idx], p_a[idx]) - average_precision_score(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    return {"delta_pr_auc": float(average_precision_score(y, p_a) - average_precision_score(y, p_b)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_of_resamples_delta_le_0": float((diffs <= 0).mean()), "n_boot": int(len(diffs))}


# ------------------------------------------------------------- behavioural tests

# (text_a, text_b): the model should score b as MORE negative than a.
NEGATION_PAIRS = [
    ("the room was clean", "the room was not clean"),
    ("the staff were helpful", "the staff were not helpful"),
    ("the bed was comfortable", "the bed was not comfortable"),
    ("breakfast was good", "breakfast was not good"),
    ("the wifi worked well", "the wifi did not work"),
    ("we would stay here again", "we would never stay here again"),
    ("i was satisfied with the room", "i was not satisfied with the room"),
    ("the hotel was worth the money", "the hotel was not worth the money"),
    ("the shower was hot", "there was no hot water in the shower"),
    ("staff were friendly", "staff were not friendly at all"),
]
# (text_a, text_b): negating a complaint should make b LESS negative than a.
NEGATED_COMPLAINT_PAIRS = [
    ("the room was dirty", "the room was not dirty"),
    ("the staff were rude", "the staff were not rude"),
    ("there were problems with the room", "there were no problems with the room"),
    ("it was bad", "it was not bad"),
    ("the room was noisy", "the room was not noisy"),
    ("i had complaints", "i had no complaints"),
    ("the bathroom was small", "the bathroom was not small"),
    ("the location was a problem", "the location was not a problem"),
]
# (text_a, text_b): only an irrelevant detail differs; |score difference| should be small.
INVARIANCE_PAIRS = [
    ("great stay in amsterdam friendly staff", "great stay in vienna friendly staff"),
    ("lovely room at the hotel arena", "lovely room at the park plaza"),
    ("the room was dirty and the staff were rude", "The room was DIRTY, and the staff were rude!!!"),
    ("breakfast was cold and expensive", "breakfast was cold and expensive we visited in june"),
    ("excellent location close to the metro", "excellent location close to the tram"),
]
INVARIANCE_TOLERANCE = 0.05


def behavioural_tests(pipe) -> dict:
    def score(t):
        return float(pipe.predict_proba([normalise(t)])[:, 1][0])

    out = {}
    for name, pairs, check in [
        ("negation: adding 'not' to praise should raise the negative score", NEGATION_PAIRS, lambda a, b: b > a),
        ("negated complaint: 'not dirty' should score less negative than 'dirty'", NEGATED_COMPLAINT_PAIRS,
         lambda a, b: b < a),
        (f"invariance: city / hotel name / punctuation should change score by < {INVARIANCE_TOLERANCE}",
         INVARIANCE_PAIRS, lambda a, b: abs(b - a) < INVARIANCE_TOLERANCE),
    ]:
        cases = []
        for a, b in pairs:
            sa, sb = score(a), score(b)
            cases.append({"a": a, "b": b, "score_a": round(sa, 3), "score_b": round(sb, 3), "pass": bool(check(sa, sb))})
        out[name] = {"pass_rate": float(np.mean([c["pass"] for c in cases])), "cases": cases}
    return out


# ------------------------------------------------------------------- subgroups

def subgroup_table(test: pd.DataFrame, p, threshold, col) -> list[dict]:
    rows = []
    for g, idx in test.groupby(col, observed=True).indices.items():
        y = test["is_negative"].to_numpy()[idx]
        if len(idx) < 300 or y.sum() < 20:
            continue
        r = {"group": str(g), **ranking_metrics(y, p[idx]), **at_threshold(y, p[idx], threshold)}
        rows.append(r)
    return rows


def add_subgroup_columns(test: pd.DataFrame) -> pd.DataFrame:
    t = test.copy()
    words = t["text"].str.split().str.len()
    t["length_band"] = pd.cut(words, [0, 10, 30, 80, 10_000], labels=["1-10 words", "11-30", "31-80", "80+"])
    if "Reviewer_Nationality" in t:
        nat = t["Reviewer_Nationality"].fillna("").str.strip()
        t["english_native_country"] = np.where(nat.isin(ENGLISH_NATIVE), "English-speaking country",
                                               "Other countries")
        top = nat.value_counts().index[:8]
        t["nationality_top8"] = np.where(nat.isin(top), nat, None)
    if "Hotel_Address" in t:
        t["city"] = t["Hotel_Address"].fillna("").map(
            lambda a: next((city for c, city in COUNTRY_TO_CITY.items() if a.strip().endswith(c)), "Other"))
    return t


# ------------------------------------------------------------------------ main

def main(argv=None) -> dict:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.DATA_PATH)
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args(argv)

    res = {}
    art = load_artifact(args.model_path)
    pipe, threshold = art["pipeline"], art["threshold"]
    C = float(pipe.named_steps["clf"].C)

    res["data_fingerprint"] = {"file": args.data.name, "sha256": sha256(args.data)}
    raw = load_raw(args.data)
    df, _ = build_dataset(raw)
    train, test = split_by_hotel(df)
    p = pipe.predict_proba(test["text"])[:, 1]
    res["primary"] = {**ranking_metrics(test["is_negative"], p), **at_threshold(test["is_negative"], p, threshold),
                      "threshold": threshold, "C": C}
    print("primary", res["primary"])

    # 2. Repeated hotel splits ---------------------------------------------------
    reps = []
    for s in args.seeds:
        tr, te = split_by_hotel(df, random_state=s)
        reps.append({"seed": s, **ranking_metrics(te["is_negative"], fit_predict(C, tr, te))})
        print("split", reps[-1])
    pr = np.array([r["pr_auc"] for r in reps])
    roc = np.array([r["roc_auc"] for r in reps])
    res["repeated_splits"] = {"runs": reps, "pr_auc_mean": float(pr.mean()), "pr_auc_sd": float(pr.std(ddof=1)),
                              "roc_auc_mean": float(roc.mean()), "roc_auc_sd": float(roc.std(ddof=1))}

    # 3. Paired significance tests -----------------------------------------------
    comparators = {
        "bigrams vs unigrams only": fit_predict(C, train, test, ngram_range=(1, 1)),
        "keeping negations vs stopwords removed": fit_predict(
            C, train, test, stop_words=list(ENGLISH_STOP_WORDS)),
    }
    current_cw = pipe.named_steps["clf"].class_weight
    alt_cw = None if current_cw == "balanced" else "balanced"
    alt = clone(pipe).set_params(clf__class_weight=alt_cw).fit(train["text"], train["is_negative"])
    label = "class weights vs no balancing" if current_cw == "balanced" else "no balancing vs class weights"
    comparators[label] = alt.predict_proba(test["text"])[:, 1]
    res["significance"] = {k: paired_bootstrap(test, p, v) for k, v in comparators.items()}
    print("significance", res["significance"])

    # 4. Label-definition sensitivity --------------------------------------------
    sens = []
    for neg_max, pos_min in [(4.0, 9.0), (5.0, 8.0), (6.0, 7.0)]:
        d, audit = build_dataset(raw, neg_max=neg_max, pos_min=pos_min)
        tr, te = split_by_hotel(d)
        pp = p if (neg_max, pos_min) == (config.NEGATIVE_MAX_SCORE, config.POSITIVE_MIN_SCORE) else fit_predict(C, tr, te)
        sens.append({"negative_if_score_le": neg_max, "positive_if_score_ge": pos_min,
                     "rows": audit["final_rows"], "excluded_as_ambiguous": audit["ambiguous_band_removed"],
                     **ranking_metrics(te["is_negative"], pp)})
        print("labels", sens[-1])
    res["label_sensitivity"] = sens

    # 5. Unseen-text subset ------------------------------------------------------
    seen = set(train["text"])
    novel = ~test["text"].isin(seen).to_numpy()
    res["unseen_text"] = {"share_of_test_with_text_seen_in_training": float(1 - novel.mean()),
                          "unseen_only": ranking_metrics(test["is_negative"].to_numpy()[novel], p[novel]),
                          "seen_only": ranking_metrics(test["is_negative"].to_numpy()[~novel], p[~novel])}

    # 6. Behavioural tests -------------------------------------------------------
    res["behavioural"] = behavioural_tests(pipe)

    # 7. Subgroups ---------------------------------------------------------------
    t = add_subgroup_columns(test)
    res["subgroups"] = {col: subgroup_table(t, p, threshold, col)
                        for col in ["length_band", "english_native_country", "nationality_top8", "city"] if col in t}

    # 8. Operating-point trade-off (thresholds chosen on out-of-fold TRAIN predictions)
    oof = cross_val_predict(clone(pipe), train["text"], train["is_negative"], groups=train["Hotel_Name"],
                            cv=GroupKFold(config.CV_FOLDS), method="predict_proba")[:, 1]
    trade = []
    y = test["is_negative"].to_numpy()
    for target in [0.70, 0.80, 0.90, 0.95]:
        thr = threshold_for_recall(train["is_negative"], oof, target)
        m = at_threshold(y, p, thr)
        per_1000_neg = 1000 * y.mean()
        trade.append({"target_recall_on_train": target, "threshold": thr, **m,
                      "per_1000_reviews": {"reviews_to_read": round(1000 * m["flag_rate"]),
                                           "unhappy_guests_in_1000": round(per_1000_neg, 1),
                                           "unhappy_caught": round(per_1000_neg * m["recall"], 1),
                                           "unhappy_missed": round(per_1000_neg * (1 - m["recall"]), 1),
                                           "reads_not_needed": round(1000 * m["flag_rate"] * (1 - m["precision"]), 1)}})
    res["operating_points"] = trade

    out = args.reports_dir
    (out / "validation.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "validation.md").write_text(to_markdown(res))
    print(to_markdown(res))
    return res


def _f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def to_markdown(r: dict) -> str:
    L = ["# Validation report (generated by `python -m hotel_sentiment.validate` - do not edit by hand)", "",
         f"Data: `{r['data_fingerprint']['file']}` · SHA-256 `{r['data_fingerprint']['sha256']}`", ""]
    pm = r["primary"]
    L += ["## 1. Primary result (held-out hotels, seed 42)", "",
          f"PR-AUC {_f(pm['pr_auc'])} · ROC-AUC {_f(pm['roc_auc'])} · at threshold {pm['threshold']:.3f}: "
          f"recall {_f(pm['recall'])}, precision {_f(pm['precision'])}, flagged {pm['flag_rate']:.1%}", ""]

    rs = r["repeated_splits"]
    L += ["## 2. Repeated hotel splits (different random sets of test hotels)", "",
          "| Seed | Test reviews | Negative rate | PR-AUC | ROC-AUC | Precision at 80% recall |", "|---|---|---|---|---|---|"]
    L += [f"| {x['seed']} | {x['n']:,} | {_f(x['negative_rate'])} | {_f(x['pr_auc'])} | {_f(x['roc_auc'])} | "
          f"{_f(x['precision_at_80_recall'])} |" for x in rs["runs"]]
    L += ["", f"**PR-AUC {rs['pr_auc_mean']:.3f} ± {rs['pr_auc_sd']:.3f} (SD) · ROC-AUC {rs['roc_auc_mean']:.3f} ± "
          f"{rs['roc_auc_sd']:.3f}**", ""]

    L += ["## 3. Are the design choices real improvements? (paired hotel bootstrap, 1,000 resamples)", "",
          "| Comparison | Δ PR-AUC | 95% CI | Share of resamples with Δ ≤ 0 |", "|---|---|---|---|"]
    for k, v in r["significance"].items():
        L.append(f"| {k} | {v['delta_pr_auc']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                 f"{v['share_of_resamples_delta_le_0']:.3f} |")
    L += [""]

    L += ["## 4. Sensitivity to the label definition", "",
          "| Negative if score ≤ | Positive if score ≥ | Reviews kept | Negative rate | PR-AUC | ROC-AUC |",
          "|---|---|---|---|---|---|"]
    L += [f"| {x['negative_if_score_le']} | {x['positive_if_score_ge']} | {x['rows']:,} | {_f(x['negative_rate'])} | "
          f"{_f(x['pr_auc'])} | {_f(x['roc_auc'])} |" for x in r["label_sensitivity"]]
    L += [""]

    u = r["unseen_text"]
    L += ["## 5. Reviews whose exact text never appears in training", "",
          f"{u['share_of_test_with_text_seen_in_training']:.1%} of test reviews have text that also appears in training "
          "(short stock phrases such as 'excellent', from other hotels).", "",
          "| Subset | Reviews | Negative rate | PR-AUC | ROC-AUC |", "|---|---|---|---|---|",
          f"| Unseen text only | {u['unseen_only']['n']:,} | {_f(u['unseen_only']['negative_rate'])} | "
          f"{_f(u['unseen_only']['pr_auc'])} | {_f(u['unseen_only']['roc_auc'])} |",
          f"| Text also in training | {u['seen_only']['n']:,} | {_f(u['seen_only']['negative_rate'])} | "
          f"{_f(u['seen_only']['pr_auc'])} | {_f(u['seen_only']['roc_auc'])} |", ""]

    L += ["## 6. Behavioural tests (minimal pairs)", ""]
    for k, v in r["behavioural"].items():
        L += [f"**{k}** — pass rate {v['pass_rate']:.0%}", "", "| A | B | score A | score B | pass |", "|---|---|---|---|---|"]
        L += [f"| {c['a']} | {c['b']} | {c['score_a']:.3f} | {c['score_b']:.3f} | {'✅' if c['pass'] else '❌'} |"
              for c in v["cases"]]
        L += [""]

    L += ["## 7. Subgroups (held-out hotels; groups with ≥ 300 reviews and ≥ 20 negatives)", "",
          "ROC-AUC is comparable across groups; PR-AUC and precision also depend on each group's negative rate.", ""]
    for col, rows in r["subgroups"].items():
        L += [f"**{col}**", "", "| Group | Reviews | Negative rate | PR-AUC | ROC-AUC | Recall @ thr | Precision @ thr |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {x['group']} | {x['n']:,} | {_f(x['negative_rate'])} | {_f(x['pr_auc'])} | {_f(x['roc_auc'])} | "
              f"{_f(x['recall'])} | {_f(x['precision'])} |" for x in rows]
        L += [""]

    L += ["## 8. Operating points: workload vs unhappy guests caught (per 1,000 reviews)", "",
          "Thresholds chosen on out-of-fold training predictions; outcomes measured on held-out hotels.", "",
          "| Target recall | Threshold | Reviews to read | Unhappy caught | Unhappy missed | Reads not needed | "
          "Test recall | Test precision |", "|---|---|---|---|---|---|---|---|"]
    for x in r["operating_points"]:
        k = x["per_1000_reviews"]
        L.append(f"| {x['target_recall_on_train']:.0%} | {x['threshold']:.3f} | {k['reviews_to_read']} | "
                 f"{k['unhappy_caught']} of {k['unhappy_guests_in_1000']} | {k['unhappy_missed']} | {k['reads_not_needed']} | "
                 f"{_f(x['recall'])} | {_f(x['precision'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
