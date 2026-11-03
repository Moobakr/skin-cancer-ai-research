# Presentation guide

## Slide 1: Title

**Explainable Multimodal Skin Cancer Screening Assistant**

Mention:

- Dataset: PAD-UFES-20
- Technologies: TensorFlow, Django, React
- Goal: educational AI screening assistant

## Slide 2: Problem

Skin lesion classification is important because early detection helps clinical care. However, visual classification is difficult, and AI models should be interpretable and safe.

## Slide 3: Dataset

Explain:

- Smartphone clinical lesion images
- Metadata file with patient and lesion information
- Multiple diagnostic classes
- Need for patient-level split

## Slide 4: Methodology

Pipeline:

```text
Dataset → preprocessing → patient-level split → CNN/transfer/multimodal training → evaluation → web deployment
```

## Slide 5: Models

Compare:

1. EfficientNetB3 image-only classifier
2. EfficientNetB3 with five metadata fields

The Custom CNN is an optional implementation and was not evaluated in this focused experiment.

## Slide 6: Evaluation

Show:

- Confusion matrix
- Accuracy/loss curves
- Classification report
- ROC-AUC if available

## Slide 7: Error analysis

Show the incorrect predictions recorded in `SCREENING_EXPERIMENT_REPORT.md`. Discuss class support, confidence errors and the limits of the comparison. Grad-CAM examples are optional future work, not part of the measured results.

## Slide 8: Web app

Show screenshots:

- Upload form
- Metadata form
- Prediction result
- Technical mode

## Slide 9: Safety features

Mention:

- Confidence levels
- Uncertainty warnings
- Image quality checks
- Medical disclaimer

## Slide 10: Conclusion

State:

> We compared image-only and five-field metadata fusion on a patient-disjoint split, documented the results and limitations, and connected the selected model to a web prototype.
