"""Verify the exported checkpoint and fill the model card from measured results."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
ARTIFACTS = ROOT / "ai_model" / "artifacts"


def main():
    selection = json.loads((ARTIFACTS / "screening_selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_model"]
    assert selected in {"image_screening", "multimodal_screening"}
    actual_hash = hashlib.sha256((BACKEND / "model_bundle" / "model.keras").read_bytes()).hexdigest()
    assert actual_hash == selection["checkpoint_sha256"][selected], "Exported checkpoint differs from validation selection"
    subprocess.run([sys.executable, "-m", "unittest", "src.test_data"], cwd=ROOT / "ai_model", check=True)
    subprocess.run([sys.executable, "manage.py", "test", "predictions"], cwd=BACKEND, check=True)

    sys.path.insert(0, str(BACKEND))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "skin_cancer_api.settings")
    import django
    django.setup()
    import pandas as pd
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT / "ai_model"))
    from src.preprocessing import decode_resize_image
    from rest_framework.test import APIRequestFactory
    from predictions.views import PredictView
    from predictions.model_service import model_service
    from predictions.preprocessing import preprocess_image_for_model

    assert model_service.model_loaded, model_service.health()
    assert not any("legacy model" in warning for warning in model_service.model_warnings()), model_service.health()
    sample = pd.read_csv(ARTIFACTS / selected / "val_split.csv").image_path.iloc[0]
    training_pixels = decode_resize_image(str(sample), 300).numpy()
    with Image.open(sample) as photo:
        api_pixels = preprocess_image_for_model(photo)[0]
    np.testing.assert_allclose(api_pixels, training_pixels, rtol=0, atol=0)
    factory = APIRequestFactory()
    for metadata in ({}, {"age": 52, "sex": "female", "anatom_site": "FACE", "smoke": "false", "drink": "true"}):
        with open(sample, "rb") as photo:
            response = PredictView.as_view()(factory.post("/api/predict/", {"image": photo, **metadata}, format="multipart"))
        assert response.status_code == 200, response.data
        assert response.data["model_mode"] == "real", response.data
        probabilities = response.data["probabilities"]
        assert set(probabilities) == set(model_service.class_names)
        assert all(0 <= value <= 1 for value in probabilities.values())
        assert abs(sum(probabilities.values()) - 1) < 1e-5
        if model_service.model_warnings():
            assert all(warning in response.data["uncertainty_warnings"] for warning in model_service.model_warnings())

    metrics = {split: json.loads((ARTIFACTS / selected / f"metrics_{split}.json").read_text(encoding="utf-8")) for split in ("val", "test")}
    melanoma = metrics["test"]["classification_report"]["MEL"]
    melanoma_support = int(melanoma["support"])
    melanoma_detected = round(melanoma["recall"] * melanoma_support)
    label = "Image + five metadata fields" if selected == "multimodal_screening" else "Image only"
    inputs = "Lesion image plus optional age, sex, body region, smoking and drinking history." if selected == "multimodal_screening" else "Lesion image. The selected image-only checkpoint does not use the optional metadata form fields."
    lines = ["# Model card: skin-lesion classification", "", "## Current model", "",
             f"**{label}** (`{selected}`), using an ImageNet-pretrained EfficientNetB3.", "",
             "Selected by validation macro-F1 before test evaluation. The deployed model matches the frozen checkpoint hash. "
             "The old model bundle is preserved in `backend/model_bundle_before_screening`.", "",
             "## Intended use", "", "Educational research prototype for PAD-UFES-20 skin-lesion classification. "
             "Not validated for diagnosis or clinical deployment.", "",
             "## Inputs and outputs", "", inputs, "",
             "Outputs: six class probabilities, predicted class, confidence and quality warnings. "
             "Softmax confidence is not calibrated. Biopsy status is excluded from the revised training inputs.", "",
             "## Measured results", "", "| Metric | Validation | Test |", "|---|---:|---:|"]
    for key, name in (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                      ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"),
                      ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted-F1")):
        lines.append(f"| {name} | {metrics['val'][key]:.3f} | {metrics['test'][key]:.3f} |")
    lines += ["", "## Validation and limits", "",
              "- Patient-disjoint splits; one dataset and one seed. No external validation or confidence intervals.",
              f"- The selected model detected {melanoma_detected}/{melanoma_support} melanoma test images. It remains a research prototype; the small cohort also makes the class-specific estimate uncertain.",
              "- API requests with supplied and blank metadata passed smoke checks. This is functionality verification, not missing-input robustness validation.",
              "- Dataset evaluation uses available metadata and a single image view. The API uses the same TensorFlow resizing operation; preprocessing parity passed for one validation PNG. This does not establish equivalence for every image format.",
              "- See [the experiment report](SCREENING_EXPERIMENT_REPORT.md) for the controlled comparison, class-level results, confusion matrices and failure examples.",
              "- Restart any already-running Django process to load the new bundle.", ""]
    (ROOT / "docs" / "MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    marker = "## Focused research revision"
    obsolete_start = text.find("## 9. What you still need to add before final submission")
    obsolete_end = text.find(marker, obsolete_start) if obsolete_start >= 0 else -1
    if obsolete_end > obsolete_start >= 0:
        text = text[:obsolete_start] + text[obsolete_end:]
    if marker in text:
        text = text[:text.index(marker)] + (
            "## Focused research revision\n\n"
            "The matched image-only and five-field metadata models have been trained and evaluated.\n"
            f"The validation-selected **{label}** model is exported to the backend.\n"
            "Read [the experiment report](docs/SCREENING_EXPERIMENT_REPORT.md) and\n"
            "[model card](docs/MODEL_CARD.md) for measured results and limitations.\n\n"
            "Biopsy status is excluded from revised training. API input values are normalized,\n"
            "image contrast is preserved, and inference uses one image view. Latency improvement\n"
            "has not been benchmarked. Restart Django to load the replacement bundle.\n\n"
            f"Melanoma detection in the test set: **{melanoma_detected}/{melanoma_support}**. "
            "The prototype should not guide clinical decisions.\n"
        )
        readme.write_text(text, encoding="utf-8")
    print("Export hash, regression tests, preprocessing parity, and real API requests with full/blank metadata verified. Model card updated.", flush=True)


if __name__ == "__main__":
    main()
