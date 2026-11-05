# Skin Cancer AI Research

A research extension examining **image-only versus image-plus-metadata skin lesion classification** with EfficientNetB3 on **PAD-UFES-20**. The study compares two classifiers on the same patient split and reports both aggregate performance and class-level failures.

[![Build and tests](https://github.com/Moobakr/skin-cancer-ai-research/actions/workflows/checks.yml/badge.svg)](https://github.com/Moobakr/skin-cancer-ai-research/actions/workflows/checks.yml)

**My project role: Model training.** This repository extends [Skin Cancer AI Assistant](https://github.com/Moobakr/skin-cancer-ai-assistant) with a focused training and evaluation study.

## Research question

**How does a classifier using lesion images and five patient fields compare with image-only EfficientNetB3 on the same patient split?**

The metadata model uses only **age, gender, body region, smoking, and drinking history**: the fields collected by the application. Biopsy status and other fields outside the form are excluded to study a more realistic input constraint.

## Experimental approach

- **Models:** ImageNet-pretrained EfficientNetB3 with 300 x 300 RGB inputs. The image-only model uses an image classification head; the multimodal model combines image features with a small metadata network.
- **Training:** Frozen-backbone training followed by fine-tuning, with augmentation, class weights, and focal loss with label smoothing. Both runs use batch size 8 and seed 42, with limits of 25 frozen-backbone epochs and 8 fine-tuning epochs.
- **Selection:** Checkpoints are chosen using validation loss. The final candidate is selected by validation macro-F1 before test evaluation.
- **Evaluation:** Single-view predictions, top-1 accuracy, balanced accuracy, macro-F1, per-class recall, and confusion matrices.

| Split | Images | Patients |
|---|---:|---:|
| Training | 1,619 | 961 |
| Validation | 337 | 206 |
| Test | 342 | 206 |

Both models use identical image membership in each split, with **zero patient overlap** between splits. Training ran on CPU.

Implementation: [model architectures](ai_model/src/models.py), [image-only training](ai_model/src/train_image_model.py), [multimodal training](ai_model/src/train_multimodal_model.py), and [evaluation](ai_model/src/evaluate.py).

## Results

| Model | Validation macro-F1 | Test accuracy | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Image only | 0.427 | 52.3% | 42.8% | 0.421 |
| Image + five metadata fields | **0.506** | **61.4%** | **49.7%** | **0.495** |

The validation-selected multimodal model scored **9.1 percentage points higher in test accuracy** and **0.074 higher in test macro-F1** than the image-only model on this split.

**Critical finding:** the multimodal model detected **0 of 5 melanoma test images**, compared with **1 of 5** for the image-only model. Better aggregate scores did not establish reliable detection of this rare class. This prototype is for educational research and has not been validated for medical diagnosis.

The full [experiment report](docs/SCREENING_EXPERIMENT_REPORT.md) includes per-class results, confusion matrices, and failure examples. [Aggregate metric files](docs/results/README.md) and the [frozen validation selection](docs/results/selection.json) record the measured results and model-selection decision.

## Interpretation and limits

- The two implemented classifiers differ in their final dropout rates, and their training included different interruptions and restarts. The observed score difference cannot be attributed solely to metadata.
- The study uses one dataset, one split, and one seed, without confidence intervals or external validation. Five melanoma test images provide limited evidence about performance on that class.
- Evaluation uses available dataset metadata. Reliability with missing form inputs and calibration of confidence scores have not been established.

The study illustrates why input availability, patient-level splitting, and class-specific evaluation matter alongside overall accuracy. Detailed limitations are recorded in the [model card](docs/MODEL_CARD.md) and experiment report.

## Application and engineering

| Component | Implementation |
|---|---|
| Model training | TensorFlow/Keras transfer learning and fine-tuning |
| Metadata processing | scikit-learn imputation, scaling, and categorical encoding |
| Inference API | Django REST Framework; normalized form values and single-view inference |
| Interface | React and Vite; image upload, optional metadata, probabilities and quality warnings |
| Input consistency | TensorFlow resizing in training evaluation and API inference; a local PNG preprocessing parity check |
| Automated checks | Patient split and feature-selection tests, backend preprocessing/inference tests, and frontend production build |

The frontend lockfile, [tested Python constraints](constraints-tested.txt), and [recorded experiment environment](docs/results/environment.json) document the software environment. CI checks software behavior; the dataset metrics were measured separately with trained models.

## Run locally

Requirements: **Python 3.12** and **Node.js 24**.

### Interface demo

A fresh clone excludes dataset images and trained weights. The following setup runs the app with **labelled placeholder predictions**.

From the repository root in PowerShell:

```powershell
py -3.12 -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -c constraints-tested.txt -r backend/requirements-core.txt
.\backend\.venv\Scripts\python.exe backend/manage.py migrate
.\backend\.venv\Scripts\python.exe backend/manage.py runserver --noreload
```

In a second terminal:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev -- --strictPort
```

Open **http://localhost:5173**. Model loading status is available at **http://localhost:8000/api/health/**. The prediction endpoint is `POST /api/predict/`.

On macOS/Linux, use `python3.12`, `backend/.venv/bin/python`, and `npm` in place of the Windows commands.

### Training and trained-model inference

Install the full runtime into the same environment:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -c constraints-tested.txt -r backend/requirements.txt
```

Follow the [dataset setup](docs/DATASET_SETUP.md) and [experiment protocol](docs/MINIMAL_EXPERIMENT_PLAN.md) to train, evaluate, and export a compatible five-field model bundle into `backend/model_bundle/`. Restart the backend after export. The recorded runs included restarts; exact weights and scores are not guaranteed by rerunning the commands.

### Software checks

```powershell
.\backend\.venv\Scripts\python.exe -m unittest ai_model.src.test_data
.\backend\.venv\Scripts\python.exe backend/manage.py test predictions
```

From `frontend`, run `npm.cmd run build`.

## Project structure

```text
ai_model/   Training, preprocessing, evaluation, and model export
backend/    Django API, inference, input validation, and tests
frontend/   React interface
docs/       Experiment protocol, results, model card, and API documentation
scripts/    Experiment reporting and local verification tools
```

See [method and dataset attribution](docs/CONTRIBUTIONS.md) and [API documentation](docs/API.md) for supporting details.
