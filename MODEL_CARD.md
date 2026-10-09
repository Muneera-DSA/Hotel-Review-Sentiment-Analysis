# Model card — Hotel review negative-sentiment triage

Format follows Mitchell et al., *Model Cards for Model Reporting* (2019). All figures come from `reports/metrics.md` and `reports/validation.md`.

## Model details

| | |
|---|---|
| Version | 2.0.0 |
| Type | Binary text classifier: TF-IDF (unigrams + bigrams, sublinear TF) → L2-regularised logistic regression (`C = 4`, no class re-weighting) |
| Artefact | `models/sentiment_pipeline.joblib`: one scikit-learn `Pipeline`, the decision threshold (0.309) and the label definition |
| Training data | 291,580 reviews from 1,193 hotels (Booking.com, 2015–2017) |
| Author | Muneera Mohamed (portfolio project, not deployed) |

## Intended use

- **Primary use:** rank incoming hotel reviews so a guest-relations team reads the likely-unhappy ones first.
- **Users:** hotel operations / guest-relations staff, with a human reading every flagged review before acting.
- **Output:** a probability that the guest was unhappy. It is well calibrated on held-out hotels, so it can be used as a probability (e.g. to forecast workload) as well as a ranking.

## Out-of-scope uses

- Automated replies, compensation or any action taken **without a human reading the review**.
- Judging or ranking **staff or individual hotels** on model scores. The scores reflect review text, and the model has not been validated as a performance measure.
- Other languages, platforms or industries without re-validation.
- Detecting fake reviews, safety incidents or legal complaints. The model was not trained or tested for these.

## Training and evaluation data

- **Source:** Kaggle *515K Hotel Reviews Data in Europe* (Booking.com, six cities). The exact file is identified by its SHA-256 fingerprint in `reports/validation.md`.
- **Label:** derived from the guest's score: negative if ≤ 5.0, positive if ≥ 8.0, scores in between excluded (149,128 reviews). These are **proxy labels**, not human annotations of the text.
- **Split:** 80/20 by hotel. The test set is 72,086 reviews from 299 hotels with no training reviews; 6.5% of test reviews are negative.

## Performance (held-out hotels)

| Metric | Value |
|---|---|
| PR-AUC | 0.849 (95% hotel-bootstrap CI 0.820–0.872); no-skill baseline 0.065 |
| ROC-AUC | 0.975 |
| At threshold 0.309 | Recall 0.809, precision 0.729, 7.2% of reviews flagged |
| Stability (5 other hotel splits) | PR-AUC 0.867 ± 0.010 |

Workload trade-off per 1,000 reviews (about 65 negative):

| Target recall | Reviews to read | Negatives caught |
|---|---|---|
| 70% | 56 | 47 |
| 80% | 72 | 53 |
| 90% | 110 | 59 |
| 95% | 169 | 62 |

## Subgroup performance

| Group | ROC-AUC | Recall at threshold |
|---|---|---|
| Reviewers from English-speaking countries | 0.977 | 0.81 |
| Reviewers from other countries | 0.971 | 0.81 |
| Cities (Amsterdam, Barcelona, London, Milan, Paris, Vienna) | 0.966–0.979 | 0.74 (Vienna) – 0.83 (Milan) |
| Reviews of 1–10 words | 0.945 | **0.64** |
| Reviews of 31+ words | 0.985–0.992 | 0.87–0.96 |

There is no material gap by reviewer origin. **Short reviews are the main weak segment.**

## Known failure modes (behavioural tests)

- **Negated complaints are often misread.** *"the room was not dirty"* scores as negative as *"the room was dirty"* (4 of 8 such tests pass). The word *not* carries strong negative weight on its own.
- **Dilution.** Appending neutral words can lower the score of a genuine complaint, because TF-IDF vectors are length-normalised (1 of 5 invariance tests fails this way).
- **Negation of praise is handled well:** *"not clean"*, *"not helpful"* and similar pass 10 of 10 tests.

## Explainability

The model provides exact SHAP values for every prediction (`python -m hotel_sentiment.explain`): which words raised or lowered a review's score, plus global importance across the test set. Reviewers should check these when a flag looks surprising.

## Ethical considerations and risks

- **Missed unhappy guests.** At the default threshold about 1 in 5 negative reviews is not flagged. Those guests are disproportionately writers of very short reviews. Teams should not treat unflagged reviews as "fine".
- **Proxy labels.** Some guests appear to misuse the rating scale (very positive text, very low score). The model learns from these noisy labels, and its reported recall is probably slightly understated.
- **Personal data.** Reviews can contain names and personal details. The model stores no review text, but any deployment should follow the operator's data-protection obligations.

## Caveats and recommendations

- Re-validate before use on later years, other platforms or other markets. The temporal split (most recent 20% of dates) showed no degradation, but it only covers 2017.
- Choose the threshold to match staff capacity using the workload table, and revisit it as volumes change.
- The next model to evaluate is a fine-tuned transformer, which can address the negation and dilution failures. It should be compared on this same hotel split and validation suite.
