"""Score new reviews with the saved pipeline.

    python -m hotel_sentiment.predict "The room was not clean and staff were rude."
    echo "Lovely stay, great breakfast" | python -m hotel_sentiment.predict
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import joblib

from . import config
from .preprocess import normalise


@lru_cache(maxsize=1)
def load_artifact(path: Path | str = config.MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Train first: python -m hotel_sentiment.train")
    return joblib.load(path)


def predict(texts: list[str], model_path: Path | str = config.MODEL_PATH) -> list[dict]:
    art = load_artifact(model_path)
    cleaned = [normalise(t) for t in texts]
    probs = art["pipeline"].predict_proba(cleaned)[:, 1]
    return [{"text": t,
             "score_negative": round(float(p), 4),
             "label": "negative" if p >= art["threshold"] else "positive"}
            for t, p in zip(texts, probs)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="*", help="review text(s); reads stdin lines if omitted")
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    args = ap.parse_args(argv)
    texts = args.texts or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        ap.error("provide at least one review")
    for r in predict(texts, args.model_path):
        print(f"{r['label']:<8}  score={r['score_negative']:.3f}  {r['text']}")


if __name__ == "__main__":
    main()
