# Skin Cancer AI: Research Extension

A research extension of the **[original Skin Cancer AI Assistant](https://github.com/Moobakr/skin-cancer-ai-assistant)**, using the **PAD-UFES-20** clinical image dataset.
This repository contains the image-only versus five-field metadata comparison.
The recovered original project and its preserved 22-field model are published separately.
It includes:

- TensorFlow/Keras training pipelines for image-only and multimodal models
- Patient-level train/validation/test splitting to reduce data leakage
- Class imbalance handling
- Grad-CAM explainability utilities
- Image quality checks
- Django REST API for predictions
- React responsive frontend
- Model card, Q&A notes, and presentation guide

> **Important medical disclaimer:** This project is for education and research only. It is not a medical diagnosis system and must not replace professional dermatology consultation.

## Repository and experiment status

This repository describes the **five-field revision**: age, gender, body region,
smoking and drinking history. The original presentation model used 22 metadata
fields and is a separate archived experiment. Its presentation figures must not
be attributed to this revision.

| Current experiment | Test top-1 accuracy | Test macro-F1 |
|---|---:|---:|
| Image only | 52.3% | 0.421 |
| Image + five metadata fields | 61.4% | 0.495 |

The metadata model was selected using validation macro-F1 before test evaluation.
It detected **0 of 5 melanoma test images**. These results do not establish clinical
screening effectiveness. See the [experiment report](docs/SCREENING_EXPERIMENT_REPORT.md),
[model card](docs/MODEL_CARD.md), and [aggregate result files](docs/results/README.md).
All reported accuracies above are single-view, top-1 scores on the held-out test set.

**Fresh clone:** datasets, trained weights, local databases and model backups are
excluded from Git. Without a model bundle, the app uses clearly labelled placeholder
demo predictions. Screenshots have not yet been added; capture them from the actual
app before including them in a portfolio. Interface screenshots are not evidence
of clinical performance.

Tested locally with **Python 3.12.2**, **Node 24.15.0**, and **npm 11.12.1**.
Use Python 3.12 and Node 24. The frontend lockfile and
[tested direct dependency constraints](constraints-tested.txt) record the package versions.

---

## 1. Project idea

Instead of building a basic image classifier, this project implements an **explainable multimodal AI assistant**:

```text
Skin lesion image + optional clinical metadata
        ↓
Django API preprocessing and model inference
        ↓
Prediction probabilities + risk level + uncertainty message + image quality warnings
        ↓
React user interface with simple and technical result views
```

The final system is designed to be easy to explain during a viva or project discussion.

---

## 2. Repository structure

```text
skin_cancer_ai_assistant/
├── ai_model/                 # Training/evaluation/export code
│   ├── src/
│   ├── artifacts/            # Generated models, metrics, class files
│   └── requirements.txt
│
├── backend/                  # Django REST API
│   ├── predictions/
│   ├── skin_cancer_api/
│   ├── manage.py
│   └── requirements.txt
│
├── frontend/                 # React + Vite frontend
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── docs/                     # Report, Q&A, model card, API docs
└── scripts/                  # Helper setup scripts
```

---

## 3. Dataset setup

Download the Kaggle dataset manually or using Kaggle API.

Expected local layout:

```text
data/pad_ufes_20/
├── metadata.csv
└── images/
    ├── PAT_1_15_820.png
    └── ...
```

The training code tries to detect the metadata file and image filename column automatically. If your extracted Kaggle folder has a different structure, set paths explicitly in the command line.

See [`docs/DATASET_SETUP.md`](docs/DATASET_SETUP.md).

---

## 4. Train the model

Use the backend Python environment created below for training as well as inference.
Install `backend/requirements.txt` with `constraints-tested.txt` before training;
the lightweight demo installation omits TensorFlow.

The recorded protocol uses EfficientNetB3, 300 by 300 images, batch size 8,
seed 42, up to 25 frozen-backbone epochs and 8 fine-tuning epochs. Exact commands
are in [the experiment plan](docs/MINIMAL_EXPERIMENT_PLAN.md).
That plan uses Windows paths; on macOS/Linux replace
`..\backend\.venv\Scripts\python.exe` with `../backend/.venv/bin/python`.

Use fresh output directories when running additional experiments. The recorded runs
included interruptions and restarts; the report documents them. Re-running training
is not a guarantee of bit-for-bit identical weights or scores.

For notebooks and plots, also install `ai_model/requirements.txt`. Matplotlib was not
used in the recorded command-line comparison and is not included in the tested
direct-runtime constraints.

---

## 5. Run Django backend

Run these commands from the repository root.

Windows PowerShell, lightweight placeholder demo:

```powershell
py -3.12 -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -c constraints-tested.txt -r backend/requirements-core.txt
.\backend\.venv\Scripts\python.exe backend/manage.py migrate
.\backend\.venv\Scripts\python.exe backend/manage.py runserver
```

macOS/Linux, lightweight placeholder demo:

```bash
python3.12 -m venv backend/.venv
backend/.venv/bin/python -m pip install -c constraints-tested.txt -r backend/requirements-core.txt
backend/.venv/bin/python backend/manage.py migrate
backend/.venv/bin/python backend/manage.py runserver
```

For real model inference or training, install the full runtime into the same environment:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -c constraints-tested.txt -r backend/requirements.txt
```

On macOS/Linux use `backend/.venv/bin/python` for that command. Full TensorFlow
installation and training depend on platform support and available hardware.
Backend configuration is read from process environment variables;
`backend/.env.example` is a reference template and is not loaded automatically.

Health check:

```text
http://localhost:8000/api/health/
```

Prediction endpoint:

```text
POST http://localhost:8000/api/predict/
```

If no trained model is exported yet, the backend runs in **demo-safe mode** and returns a clear warning that predictions are placeholders.

---

## 6. Run React frontend

```bash
cd frontend
npm ci
npm run dev
```

Open:

```text
http://localhost:5173
```

In PowerShell, use `npm.cmd` if execution policy prevents running `npm.ps1`.
The optional `frontend/.env.example` documents the API URL override.

---

## 7. Best features to present

1. **Patient-level split** to reduce data leakage.
2. **Transfer learning** using EfficientNetB3.
3. **Multimodal model** combining image and metadata.
4. **Confidence-aware prediction** with uncertainty warnings.
5. **Risk grouping** for user-friendly output.
6. **Image quality checks** for brightness, size, and blur.
7. **Grad-CAM utility** for explainability.
8. **Full-stack Django + React integration**.

---

## 8. Contributions, checks and publishing

See [contributions and attribution](docs/CONTRIBUTIONS.md) for method provenance,
AI-assistance disclosure and guidance for assigning individual/team credit.
The architecture uses established methods; the experiment is a comparison of two
implemented classifiers, not a claim of a novel medical diagnostic method.

Run the lightweight checks with the backend environment:

```powershell
.\backend\.venv\Scripts\python.exe backend/manage.py test predictions
.\backend\.venv\Scripts\python.exe -m unittest ai_model.src.test_data
```

From `frontend`, run `npm ci` and `npm run build`. GitHub Actions runs these backend
and data tests plus the frontend production build with Python 3.12 and Node 24.
CI does not download patient data or weights and does not validate model accuracy.
The full local bundle verification script requires the trained bundle and dataset.

Initialize Git in this inner `skin_cancer_ai_assistant` directory. Keep datasets,
virtual environments, databases, model backups and checkpoints out of commits.
If weights are distributed separately, first review their applicable terms and
include a complete compatible bundle. The repository itself does not provide a
trained-weight download.

**License:** no source-code license has been selected, at the owner's request.
Dataset, pretrained-weight and dependency terms remain separate.

---

## Focused research revision

The matched image-only and five-field metadata models have been trained and evaluated.
The validation-selected **Image + five metadata fields** model is exported to the backend.
Read [the experiment report](docs/SCREENING_EXPERIMENT_REPORT.md) and
[model card](docs/MODEL_CARD.md) for measured results and limitations.

Biopsy status is excluded from revised training. API input values are normalized,
image contrast is preserved, and inference uses one image view. Latency improvement
has not been benchmarked. Restart Django to load the replacement bundle.

Melanoma detection in the test set: **0/5**. The prototype should not guide clinical decisions.
