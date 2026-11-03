# Focused skin-lesion classification experiment

## Research question

How does a classifier combining images with age, gender, body region, smoking and drinking history compare with image-only EfficientNetB3 on an identical patient split?

## Method

[PAD-UFES-20](https://arxiv.org/abs/2007.00478); six classes; ImageNet-pretrained [EfficientNetB3](https://arxiv.org/abs/1905.11946); 300 × 300 RGB inputs; batch size 8; seed 42. Both models use the same split, augmentation, class weights, focal loss with label smoothing, and learning-rate schedule. The metadata model adds a small fusion branch. Biopsy status and all fields absent from the app are excluded.

Training allows up to 25 frozen-backbone epochs and eight fine-tuning epochs, with early stopping. Checkpoints are chosen using validation loss. The deployed candidate is chosen by validation macro-F1 before reading test results. Evaluation uses a single view. Training ran on CPU.

| Split | Images | Patients |
|---|---:|---:|
| train | 1619 | 961 |
| val | 337 | 206 |
| test | 342 | 206 |

Patient overlap is zero; both models use the same image membership in every split.

## Results

| Model | Split | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Image only | val | 0.543 | 0.459 | 0.427 |
| Image only | test | 0.523 | 0.428 | 0.421 |
| Image + five metadata fields | val | 0.573 | 0.501 | 0.506 |
| Image + five metadata fields | test | 0.614 | 0.497 | 0.495 |

Validation-selected candidate: **Image + five metadata fields**. The test macro-F1 difference (multimodal minus image-only) is **+0.074**. This is one observed split, not evidence of a statistically established improvement.

**Critical finding:** the selected model detected **0/5 melanoma test images**. The higher aggregate macro-F1 must not be interpreted as evidence of effective clinical screening. The model choice remains the validation-selected choice; test results were not used to switch candidates.

### Per-class test recall

| Class | Test images | Image-only recall | Multimodal recall |
|---|---:|---:|---:|
| ACK | 100 | 0.740 | 0.760 |
| BCC | 112 | 0.509 | 0.580 |
| MEL | 5 | 0.200 | 0.000 |
| NEV | 42 | 0.690 | 0.833 |
| SCC | 40 | 0.150 | 0.250 |
| SEK | 43 | 0.279 | 0.558 |

### Test confusion matrices

Rows are true classes; columns are predicted classes.

**Image only**

| True / predicted | ACK | BCC | MEL | NEV | SCC | SEK |
|---|---:|---:|---:|---:|---:|---:|
| ACK | 74 | 17 | 1 | 1 | 7 | 0 |
| BCC | 30 | 57 | 3 | 8 | 14 | 0 |
| MEL | 3 | 1 | 1 | 0 | 0 | 0 |
| NEV | 6 | 4 | 2 | 29 | 0 | 1 |
| SCC | 14 | 19 | 0 | 0 | 6 | 1 |
| SEK | 12 | 6 | 5 | 8 | 0 | 12 |

**Image + five metadata fields**

| True / predicted | ACK | BCC | MEL | NEV | SCC | SEK |
|---|---:|---:|---:|---:|---:|---:|
| ACK | 76 | 10 | 0 | 2 | 9 | 3 |
| BCC | 22 | 65 | 0 | 5 | 17 | 3 |
| MEL | 3 | 2 | 0 | 0 | 0 | 0 |
| NEV | 4 | 1 | 0 | 35 | 0 | 2 |
| SCC | 10 | 20 | 0 | 0 | 10 | 0 |
| SEK | 8 | 5 | 0 | 6 | 0 | 24 |

### Failure examples from the selected model

Highest-confidence incorrect predictions; confidence is not calibrated.

| Image | True class | Predicted class | Confidence |
|---|---|---|---:|
| PAT_380_1540_432.png | SCC | ACK | 0.746 |
| PAT_380_773_542.png | BCC | ACK | 0.727 |
| PAT_489_932_138.png | BCC | NEV | 0.718 |

## Limits and interpretation

- One random seed and one dataset; no external validation or confidence intervals.
- Earlier batch-16 runs were archived after memory pressure. Both final runs use batch size 8, bounded image loading/prefetching, four intra-op CPU threads and one inter-op thread. Training times are not a controlled speed comparison.
- The batch-8 image run was interrupted after its frozen stage and five fine-tuning epochs. Its fine-tuning phase was restarted from the saved frozen checkpoint; this is not an exact optimizer/RNG resume. The original and recovery logs are retained. This limits attribution of differences solely to metadata.
- Metadata training was resumed from an epoch-8 backup. Weights, optimizer and epoch position are restored, but exact data-order/RNG and callback-state continuity is not guaranteed.
- The fusion head uses dropout 0.35 after its dense layer; the image-only head uses 0.25. This is a comparison of the two implemented classifiers, not a pure metadata-only ablation; regularization differences also limit causal attribution.
- Only five melanoma images occur in the test set; one mistake changes melanoma recall by 20 percentage points.
- Dataset evaluation uses available metadata. Blank form inputs are accepted and imputed, but missing-input reliability is not established.
- Softmax scores and Grad-CAM visualizations do not establish clinical reliability or causal explanations.
- The API and evaluation use TensorFlow resizing. A validation-PNG preprocessing parity check runs after export; other image formats have not been checked for exact pixel equivalence.
- The architecture is established. The contribution is a controlled implementation and evaluation with realistic feature restrictions.
- This prototype has not been validated for clinical use.

## Reproduction and application use

Commands are in `MINIMAL_EXPERIMENT_PLAN.md`. Exact environment versions and selection rules are saved in `ai_model/artifacts/screening_experiment.json`; model configs, training logs, split CSVs, metrics and sample-level predictions are in each model's artifact directory.

For an application, explain the motivation, leakage correction, controlled comparison, and observed failures. Describe personal contributions accurately, including tools or assistance used. Present this as a reproducible student research project; do not claim novelty or clinical readiness.

## References

- Pacheco et al. (2020). [PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones](https://arxiv.org/abs/2007.00478).
- Tan and Le (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
