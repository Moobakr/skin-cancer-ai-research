# Implementation checklist

Use this before submission.

This is a manual review template. The focused experiment does not require a Custom CNN comparison or Grad-CAM examples. Use `SCREENING_EXPERIMENT_REPORT.md` and the run status JSON for measured completion; do not claim unchecked manual UI checks as verified.

## Dataset and training

- [ ] Dataset downloaded into `data/pad_ufes_20/`
- [ ] Metadata CSV detected correctly
- [ ] Patient-level train/validation/test split confirmed
- [ ] Custom CNN trained (optional, outside the focused experiment)
- [ ] EfficientNetB3 image model trained
- [ ] Multimodal model trained if metadata features are available
- [ ] Best model exported to `backend/model_bundle/model.keras`
- [ ] `class_names.json` exists in backend model bundle
- [ ] `metadata_preprocessor.joblib` and `metadata_schema.json` exported for multimodal model

## Evaluation

- [ ] Accuracy reported
- [ ] Macro precision/recall/F1 reported
- [ ] Weighted F1 reported
- [ ] Confusion matrix created
- [ ] At least 3 validated Grad-CAM examples generated (optional)
- [ ] Limitations written honestly

## Backend

- [ ] `GET /api/health/` works
- [ ] `POST /api/predict/` works with one image
- [ ] Demo mode is disabled after real model export
- [ ] Image quality warnings work
- [ ] Medical disclaimer is returned

## Frontend

- [ ] User can upload image
- [ ] Preview appears
- [ ] Metadata fields work
- [ ] Loading state appears
- [ ] Result card appears
- [ ] Simple/technical mode switch works
- [ ] Print report button works
- [ ] Mobile layout looks good

## Presentation

- [ ] Explain why patient-level split matters
- [ ] Explain CNN/transfer learning clearly
- [ ] Explain multimodal fusion
- [ ] Explain confidence and uncertainty warnings
- [ ] Explain why this is not a medical diagnosis
