import numpy as np
import pytest

from hotel_sentiment.explain import LinearShap
from hotel_sentiment.model import build_pipeline
from synthetic import make_reviews


def _fitted():
    df = make_reviews(1200, seed=3)
    texts = (df["Positive_Review"] + " " + df["Negative_Review"]).str.lower().tolist()
    y = (df["Reviewer_Score"] <= 5.0).astype(int).to_numpy()
    pipe = build_pipeline(min_df=1).fit(texts, y)
    return pipe, texts


def test_shap_values_sum_to_model_output():
    pipe, texts = _fitted()
    ex = LinearShap(pipe, texts)
    X = ex.transform(texts[:50])
    phi = ex.shap_dense(X)
    np.testing.assert_allclose(ex.base_value + phi.sum(axis=1), pipe.decision_function(texts[:50]), atol=1e-9)


def test_explain_one_matches_dense():
    pipe, texts = _fitted()
    ex = LinearShap(pipe, texts)
    X = ex.transform(texts[:5])
    for i in range(5):
        e = ex.explain_one(X[i], top=3)
        assert e["logit"] == pytest.approx(pipe.decision_function([texts[i]])[0], abs=1e-9)


def test_global_importance_matches_dense():
    pipe, texts = _fitted()
    ex = LinearShap(pipe, texts)
    X = ex.transform(texts[:200])
    imp = ex.global_importance(X, top=5)
    dense = np.abs(ex.shap_dense(X)).mean(axis=0)
    expected = ex.vocab[np.argsort(-dense)[:5]]
    assert list(imp["term"]) == list(expected)
    np.testing.assert_allclose(imp["mean_abs_shap"], np.sort(dense)[::-1][:5], rtol=1e-9)


def test_matches_shap_library():
    shap = pytest.importorskip("shap")
    pipe, texts = _fitted()
    ex = LinearShap(pipe, texts)
    Xbg = ex.transform(texts)
    X = ex.transform(texts[:20])
    # shap's default masker subsamples the background to 100 rows; use the full
    # background so both implementations use the same E[x].
    Xbg, X = Xbg.toarray(), X.toarray()
    masker = shap.maskers.Independent(Xbg, max_samples=Xbg.shape[0])
    ref = np.asarray(shap.LinearExplainer(pipe.named_steps["clf"], masker).shap_values(X))
    np.testing.assert_allclose(ex.shap_dense(ex.transform(texts[:20])), ref, atol=1e-8)
