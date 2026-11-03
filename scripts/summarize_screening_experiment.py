"""Build the short comparison report from measured results only.

Run with the backend Python environment after both models have been evaluated.
Requires screening_selection.json to record the validation-only model choice.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ai_model" / "artifacts"
MODELS = {"image_screening": "Image only", "multimodal_screening": "Image + five metadata fields"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    selection = read_json(ARTIFACTS / "screening_selection.json")
    configs = {name: read_json(ARTIFACTS / name / "training_config.json") for name in MODELS}
    for config in configs.values():
        assert config["batch_size"] == 8 and config["seed"] == 42 and config["image_size"] == 300
        assert config["epochs"] == 25 and config["fine_tune_epochs"] == 8
    results = {
        model: {split: read_json(ARTIFACTS / model / f"metrics_{split}.json") for split in ("val", "test")}
        for model in MODELS
    }
    classes = results["image_screening"]["test"]["class_names"]
    split_rows = {
        split: read_csv(ARTIFACTS / "image_screening" / f"{split}_split.csv")
        for split in ("train", "val", "test")
    }
    groups = {split: {row["patient_id"] for row in rows} for split, rows in split_rows.items()}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
    for split, rows in split_rows.items():
        other = read_csv(ARTIFACTS / "multimodal_screening" / f"{split}_split.csv")
        assert {row["img_id"] for row in rows} == {row["img_id"] for row in other}
    assert selection["selected_model"] in MODELS
    assert selection["selected_model"] == max(MODELS, key=lambda name: results[name]["val"]["macro_f1"])
    for model in MODELS:
        for split in ("val", "test"):
            assert not results[model][split]["legacy_biopsy_input"]
    selected = selection["selected_model"]
    lines = [
        "# Focused skin-lesion classification experiment", "",
        "## Research question", "",
        "How does a classifier combining images with age, gender, body region, smoking "
        "and drinking history compare with image-only EfficientNetB3 on an identical patient split?", "",
        "## Method", "",
        "[PAD-UFES-20](https://arxiv.org/abs/2007.00478); six classes; ImageNet-pretrained "
        "[EfficientNetB3](https://arxiv.org/abs/1905.11946); 300 × 300 RGB inputs; "
        "batch size 8; seed 42. Both models use the same split, augmentation, class weights, "
        "focal loss with label smoothing, and learning-rate schedule. The metadata model "
        "adds a small fusion branch. Biopsy status and all fields absent from the app are excluded.", "",
        "Training allows up to 25 frozen-backbone epochs and eight fine-tuning epochs, "
        "with early stopping. Checkpoints are chosen using validation loss. The deployed "
        "candidate is chosen by validation macro-F1 before reading test results. "
        "Evaluation uses a single view. Training ran on CPU.", "",
        "| Split | Images | Patients |", "|---|---:|---:|",
    ]
    for split, rows in split_rows.items():
        lines.append(f"| {split} | {len(rows)} | {len(groups[split])} |")
    lines += ["", "Patient overlap is zero; both models use the same image membership in every split.", "",
              "## Results", "", "| Model | Split | Accuracy | Balanced accuracy | Macro-F1 |",
              "|---|---|---:|---:|---:|"]
    for model, label in MODELS.items():
        for split in ("val", "test"):
            metrics = results[model][split]
            lines.append(f"| {label} | {split} | {metrics['accuracy']:.3f} | {metrics['balanced_accuracy']:.3f} | {metrics['macro_f1']:.3f} |")
    difference = results["multimodal_screening"]["test"]["macro_f1"] - results["image_screening"]["test"]["macro_f1"]
    melanoma = results[selected]["test"]["classification_report"]["MEL"]
    melanoma_support = int(melanoma["support"])
    melanoma_detected = round(melanoma["recall"] * melanoma_support)
    lines += ["", f"Validation-selected candidate: **{MODELS[selected]}**. "
              f"The test macro-F1 difference (multimodal minus image-only) is **{difference:+.3f}**. "
              "This is one observed split, not evidence of a statistically established improvement.", "",
              f"**Critical finding:** the selected model detected **{melanoma_detected}/{melanoma_support} melanoma test images**. "
              "The higher aggregate macro-F1 must not be interpreted as evidence of effective clinical screening. "
              "The model choice remains the validation-selected choice; test results were not used to switch candidates.", "",
              "### Per-class test recall", "", "| Class | Test images | Image-only recall | Multimodal recall |",
              "|---|---:|---:|---:|"]
    for name in classes:
        image = results["image_screening"]["test"]["classification_report"][name]
        multi = results["multimodal_screening"]["test"]["classification_report"][name]
        lines.append(f"| {name} | {int(image['support'])} | {image['recall']:.3f} | {multi['recall']:.3f} |")
    lines += ["", "### Test confusion matrices", "", "Rows are true classes; columns are predicted classes."]
    for model, label in MODELS.items():
        lines += ["", f"**{label}**", "", "| True / predicted | " + " | ".join(classes) + " |",
                  "|---|" + "---:|" * len(classes)]
        for name, row in zip(classes, results[model]["test"]["confusion_matrix"]):
            lines.append("| " + name + " | " + " | ".join(map(str, row)) + " |")
    lines += ["", "### Failure examples from the selected model", "",
              "Highest-confidence incorrect predictions; confidence is not calibrated.", "",
              "| Image | True class | Predicted class | Confidence |", "|---|---|---|---:|"]
    predictions = read_csv(ARTIFACTS / selected / "predictions_test.csv")
    errors = sorted((row for row in predictions if row["true_class"] != row["predicted_class"]),
                    key=lambda row: float(row["confidence"]), reverse=True)
    for row in errors[:3]:
        lines.append(f"| {row['img_id']} | {row['true_class']} | {row['predicted_class']} | {float(row['confidence']):.3f} |")
    lines += ["", "## Limits and interpretation", "",
              "- One random seed and one dataset; no external validation or confidence intervals.",
              "- Earlier batch-16 runs were archived after memory pressure. Both final runs use batch size 8, bounded image loading/prefetching, four intra-op CPU threads and one inter-op thread. Training times are not a controlled speed comparison.",
              "- The batch-8 image run was interrupted after its frozen stage and five fine-tuning epochs. Its fine-tuning phase was restarted from the saved frozen checkpoint; this is not an exact optimizer/RNG resume. The original and recovery logs are retained. This limits attribution of differences solely to metadata.",
              "- Metadata training was resumed from an epoch-8 backup. Weights, optimizer and epoch position are restored, but exact data-order/RNG and callback-state continuity is not guaranteed.",
              "- The fusion head uses dropout 0.35 after its dense layer; the image-only head uses 0.25. This is a comparison of the two implemented classifiers, not a pure metadata-only ablation; regularization differences also limit causal attribution.",
              "- Only five melanoma images occur in the test set; one mistake changes melanoma recall by 20 percentage points.",
              "- Dataset evaluation uses available metadata. Blank form inputs are accepted and imputed, but missing-input reliability is not established.",
              "- Softmax scores and Grad-CAM visualizations do not establish clinical reliability or causal explanations.",
              "- The API and evaluation use TensorFlow resizing. A validation-PNG preprocessing parity check runs after export; other image formats have not been checked for exact pixel equivalence.",
              "- The architecture is established. The contribution is a controlled implementation and evaluation with realistic feature restrictions.",
              "- This prototype has not been validated for clinical use.", "",
              "## Reproduction and application use", "",
              "Commands are in `MINIMAL_EXPERIMENT_PLAN.md`. Exact environment versions and selection rules "
              "are saved in `ai_model/artifacts/screening_experiment.json`; model configs, training logs, "
              "split CSVs, metrics and sample-level predictions are in each model's artifact directory.", "",
              "For an application, explain the motivation, leakage correction, controlled comparison, "
              "and observed failures. Describe personal contributions accurately, including tools or assistance "
              "used. Present this as a reproducible student research project; do not claim novelty or clinical readiness.", "",
              "## References", "",
              "- Pacheco et al. (2020). [PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones](https://arxiv.org/abs/2007.00478).",
              "- Tan and Le (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).", ""]
    output = ROOT / "docs" / "SCREENING_EXPERIMENT_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
