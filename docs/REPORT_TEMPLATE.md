# Report Template

This is an optional blank template. The measured two-model comparison is generated in `SCREENING_EXPERIMENT_REPORT.md`. Report only experiments actually performed.

## 1. Abstract

This project develops an explainable multimodal deep learning web application for skin lesion classification using the PAD-UFES-20 dataset. The system combines lesion image analysis with optional clinical metadata and deploys the trained model through a Django REST API and responsive React frontend.

## 2. Introduction

Explain the motivation for skin lesion classification and why explainable AI is important in medical applications.

## 3. Dataset

Describe the dataset, diagnostic classes, images, metadata, and any imbalance observed during exploratory analysis.

## 4. Preprocessing

Include:

- Image resizing
- Normalization/preprocessing
- Data augmentation
- Metadata encoding/scaling
- Patient-level split

## 5. Model Architecture

Describe:

- Custom CNN (optional, not run in the focused comparison)
- EfficientNetB3 transfer learning
- Multimodal fusion model

## 6. Training Strategy

Include:

- Loss function
- Optimizer
- Class weights
- Early stopping
- Learning-rate reduction
- Fine-tuning

## 7. Evaluation

Add final metrics after training.

| Model | Accuracy | Macro Recall | Macro F1 | Notes |
|---|---:|---:|---:|---|
| Custom CNN | Not run | Not run | Not run | Optional baseline |
| EfficientNetB3 | TODO | TODO | TODO | Image-only candidate |
| EfficientNetB3 + Metadata | TODO | TODO | TODO | Fusion candidate |

## 8. Explainability

Include validated Grad-CAM examples only if separately generated and checked. Otherwise use the error analysis from the measured report and identify heatmaps as optional future work.

## 9. Web Application

Explain Django backend and React frontend architecture.

## 10. Limitations

Mention dataset size, class imbalance, non-clinical validation, and the need for dermatologist consultation.

## 11. Conclusion

Summarize the full pipeline and unique project contributions.
