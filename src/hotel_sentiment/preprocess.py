"""Text normalisation.

Design decision: no stopword removal and no lemmatisation.
Standard English stopword lists (e.g. NLTK) contain "not", "no", "nor",
"don't", "wasn't". Removing them turns "the room was not clean" into
"room clean" and flips the meaning. TF-IDF with bigrams and sublinear term
frequency already down-weights uninformative frequent words, so stopword
removal adds risk without adding signal. This also removes the NLTK
dependency and its runtime downloads.
"""

import re

from .config import PLACEHOLDERS

_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(p) for p in PLACEHOLDERS) + r")\s*$",
    flags=re.IGNORECASE,
)
# Expand the common English contractions so "wasn't" and "was not" share features.
_CONTRACTIONS = {
    r"\bcan't\b": "can not",
    r"\bwon't\b": "will not",
    r"n't\b": " not",
}
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_placeholder(field: str) -> str:
    """Return '' if a review field is only a Booking.com placeholder, else the field."""
    if not isinstance(field, str):
        return ""
    return "" if _PLACEHOLDER_RE.match(field) else field.strip()


def normalise(text: str) -> str:
    """Lowercase, expand negated contractions, keep letters only, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("’", "'")
    for pattern, repl in _CONTRACTIONS.items():
        text = re.sub(pattern, repl, text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
