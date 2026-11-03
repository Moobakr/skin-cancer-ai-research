# Minimal experiment plan

Question: How does a classifier using images and the five patient fields collected
by the app compare with image-only EfficientNetB3 on the same patient split?

Scope: two models, one dataset, one fixed seed, the existing web app. No new
architectures, datasets, or deployment platform are needed for this revision.
This is a controlled student project, not a claim of methodological novelty.

## Correctness changes

- New multimodal training allows only age, gender, region, smoke and drink.
  Biopsy status and other diagnostic-process fields are excluded.
- API values match training categories; missing values use the fitted imputer.
- The API no longer applies contrast enhancement absent from training.
- Inference uses one image rather than five transformed views. This reduces work;
  no latency improvement or accuracy preservation is claimed without measurement.
- Evaluation loads inference-only models without deserializing the custom loss,
  preserves all six classes in reports, and records balanced accuracy and support.

## Run the two experiments

From the project root in PowerShell:

```powershell
cd ai_model
$env:TF_NUM_INTRAOP_THREADS = '4'
$env:TF_NUM_INTEROP_THREADS = '1'
..\backend\.venv\Scripts\python.exe -m src.train_image_model --dataset-dir ../data/pad_ufes_20 --output-dir artifacts/image_screening --model efficientnet --epochs 25 --fine-tune-epochs 8 --batch-size 8 --seed 42
..\backend\.venv\Scripts\python.exe -m src.train_multimodal_model --dataset-dir ../data/pad_ufes_20 --output-dir artifacts/multimodal_screening --epochs 25 --fine-tune-epochs 8 --batch-size 8 --seed 42
```

These write new artifact directories and preserve the existing model. Both use the
same patient split, epoch ceilings and class weighting. Extra oversampling was
removed from multimodal training so the balancing procedures match.

The existing heads retain different final dropout rates (image-only 0.25; fusion
0.35). Treat the results as a comparison of these two classifiers, with this
regularization difference disclosed, rather than a pure metadata-only ablation.

Both final models use batch size 8, two parallel image loads, one prefetched batch,
and TensorFlow environment settings TF_NUM_INTRAOP_THREADS=4 and
TF_NUM_INTEROP_THREADS=1. Earlier batch-16 runs were archived after memory pressure
and are excluded from the final comparison. Details are recorded in
artifacts/screening_experiment.json. Training speeds are not compared.

Evaluate validation first:

```powershell
..\backend\.venv\Scripts\python.exe -m src.evaluate --trained-dir artifacts/image_screening --split val
..\backend\.venv\Scripts\python.exe -m src.evaluate --trained-dir artifacts/multimodal_screening --split val
```

Freeze choices before evaluating test:

```powershell
..\backend\.venv\Scripts\python.exe -m src.evaluate --trained-dir artifacts/image_screening --split test
..\backend\.venv\Scripts\python.exe -m src.evaluate --trained-dir artifacts/multimodal_screening --split test
```

Report accuracy, balanced accuracy, macro-F1, per-class recall and support. Explain
that one split gives limited evidence, particularly for rare classes. Inspect the
saved patient IDs for overlap and class counts before interpreting the results.
The dataset evaluation uses available metadata; it does not establish robustness
to missing app inputs. Do not tune on test results.

Export the revised multimodal model only after reviewing its results:

```powershell
..\backend\.venv\Scripts\python.exe -m src.export_bundle --trained-dir artifacts/multimodal_screening --backend-bundle-dir ../backend/model_bundle
```

Restart Django after export. This overwrites the deployed bundle, while the old
training artifacts remain in `artifacts/multimodal_model`.

## Application evidence

Prepare a short report with the question, split method, two-model comparison,
confusion matrix, several failure examples, limitations, and your personal
contribution. Include a short app demonstration. Use measured results only; a
well-explained negative result is still useful evidence of careful work.

## Checks

From `ai_model`: `..\backend\.venv\Scripts\python.exe -m unittest src.test_data`

From `backend`: `.\.venv\Scripts\python.exe manage.py test predictions`

From `frontend`: `npm.cmd run build`
