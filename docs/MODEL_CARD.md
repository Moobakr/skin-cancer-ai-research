# Model card: skin-lesion classification

## Current model

**Image + five metadata fields** (`multimodal_screening`), using an ImageNet-pretrained EfficientNetB3.

Selected by validation macro-F1 before test evaluation. The deployed model matches the frozen checkpoint hash. The old model bundle is preserved in `backend/model_bundle_before_screening`.

## Intended use

Educational research prototype for PAD-UFES-20 skin-lesion classification. Not validated for diagnosis or clinical deployment.

## Inputs and outputs

Lesion image plus optional age, sex, body region, smoking and drinking history.

Outputs: six class probabilities, predicted class, confidence and quality warnings. Softmax confidence is not calibrated. Biopsy status is excluded from the revised training inputs.

## Measured results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.573 | 0.614 |
| Balanced accuracy | 0.501 | 0.497 |
| Macro precision | 0.529 | 0.501 |
| Macro recall | 0.501 | 0.497 |
| Macro-F1 | 0.506 | 0.495 |
| Weighted-F1 | 0.574 | 0.604 |

## Validation and limits

- Patient-disjoint splits; one dataset and one seed. No external validation or confidence intervals.
- The selected model detected 0/5 melanoma test images. It remains a research prototype; the small cohort also makes the class-specific estimate uncertain.
- API requests with supplied and blank metadata passed smoke checks. This is functionality verification, not missing-input robustness validation.
- Dataset evaluation uses available metadata and a single image view. The API uses the same TensorFlow resizing operation; preprocessing parity passed for one validation PNG. This does not establish equivalence for every image format.
- See [the experiment report](SCREENING_EXPERIMENT_REPORT.md) for the controlled comparison, class-level results, confusion matrices and failure examples.
- Restart any already-running Django process to load the new bundle.
