# Aggregate results: five-field revision

These JSON files are copies of the measured aggregate evaluation outputs. They
contain class-level metrics and confusion matrices, not patient rows, images,
absolute filesystem paths or per-image predictions.

| File | Meaning |
|---|---|
| `image_screening_val.json` | Image-only validation results |
| `image_screening_test.json` | Image-only held-out test results |
| `multimodal_screening_val.json` | Five-field fusion validation results |
| `multimodal_screening_test.json` | Five-field fusion held-out test results |
| `selection.json` | Validation-only selection rule, scores and checkpoint hashes |
| `environment.json` | Recorded direct dependency versions and experiment settings |

Accuracy is **top-1**, using one image view per prediction. The candidate was
selected by validation macro-F1 before evaluating the held-out test set. The
patient split contains 1,619 training images, 337 validation images and 342 test
images, with no patient overlap.

The selected five-field model missed all five melanoma test images. Aggregate
performance does not establish effective clinical screening.

These files describe the five-field experiment only. The original 22-field
presentation model is a separate local archive; these numbers must not be used
as its presentation results. Exact training restarts, head differences and other
limitations are documented in the [experiment report](../SCREENING_EXPERIMENT_REPORT.md).

The [model card](../MODEL_CARD.md) describes intended use. Source dataset and
framework attribution is in [CONTRIBUTIONS.md](../CONTRIBUTIONS.md).
