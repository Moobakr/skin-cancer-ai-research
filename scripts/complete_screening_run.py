"""Finish the authorized comparison once the active metadata training run ends.

Run with backend/.venv/Scripts/python.exe. Progress is written to
ai_model/artifacts/screening_run_status.json. Existing evaluated test results are
reused only when the frozen checkpoint hashes match.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AI = ROOT / "ai_model"
ARTIFACTS = AI / "artifacts"
MODELS = ("image_screening", "multimodal_screening")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def status(stage, **details):
    value = {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **details}
    write(ARTIFACTS / "screening_run_status.json", value)
    print(json.dumps(value), flush=True)


def run(*args):
    subprocess.run([sys.executable, *args], cwd=AI, check=True)


def main():
    status("waiting_for_metadata_training")
    deadline = time.monotonic() + 24 * 60 * 60
    completed = ARTIFACTS / "multimodal_screening" / "training_config.json"
    while not completed.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Metadata training has not completed within 24 hours. Rerun this completion script after training ends.")
        time.sleep(10)

    schema = read(completed.parent / "metadata_schema.json")
    if schema["feature_columns"] != ["age", "gender", "region", "smoke", "drink"]:
        raise ValueError("Unexpected metadata schema; refusing to deploy.")
    fingerprints = {
        model: hashlib.sha256((ARTIFACTS / model / "best_model.keras").read_bytes()).hexdigest()
        for model in MODELS
    }
    selection_path = ARTIFACTS / "screening_selection.json"
    if selection_path.exists() and read(selection_path).get("checkpoint_sha256") != fingerprints:
        raise ValueError("Checkpoints changed after selection. Preserve the previous experiment before rerunning.")

    for model in MODELS:
        status("validation", model=model)
        metrics = ARTIFACTS / model / "metrics_val.json"
        if not metrics.exists():
            run("-m", "src.evaluate", "--trained-dir", f"artifacts/{model}", "--split", "val", "--batch-size", "8")
        elif metrics.stat().st_mtime < (ARTIFACTS / model / "best_model.keras").stat().st_mtime:
            raise ValueError(f"Stale validation metrics: {model}")

    scores = {model: read(ARTIFACTS / model / "metrics_val.json")["macro_f1"] for model in MODELS}
    selected = max(MODELS, key=lambda model: scores[model])
    if not selection_path.exists():
        if any((ARTIFACTS / model / "metrics_test.json").exists() for model in MODELS):
            raise ValueError("Test metrics exist without a frozen validation selection.")
        write(selection_path, {
            "selected_model": selected, "criterion": "validation macro_f1",
            "validation_scores": scores, "checkpoint_sha256": fingerprints,
            "frozen_before_test_evaluation_utc": datetime.now(timezone.utc).isoformat(),
        })
    status("selection_frozen", selected_model=selected, validation_scores=scores)

    for model in MODELS:
        status("test_evaluation", model=model)
        if not (ARTIFACTS / model / "metrics_test.json").exists():
            run("-m", "src.evaluate", "--trained-dir", f"artifacts/{model}", "--split", "test", "--batch-size", "8")

    status("writing_report")
    run(str(ROOT / "scripts" / "summarize_screening_experiment.py"))
    bundle = ROOT / "backend" / "model_bundle"
    backup = ROOT / "backend" / "model_bundle_before_screening"
    if bundle.exists() and not backup.exists():
        shutil.copytree(bundle, backup)
    status("exporting", selected_model=selected)
    run("-m", "src.export_bundle", "--trained-dir", f"artifacts/{selected}", "--backend-bundle-dir", str(bundle))
    status("exported_needs_final_verification", selected_model=selected,
           report=str(ROOT / "docs" / "SCREENING_EXPERIMENT_REPORT.md"))
    run(str(ROOT / "scripts" / "verify_screening_run.py"))
    status("complete", selected_model=selected,
           report=str(ROOT / "docs" / "SCREENING_EXPERIMENT_REPORT.md"),
           test_metrics=read(ARTIFACTS / selected / "metrics_test.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=str(error))
        raise
