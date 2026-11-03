from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained model files to Django model_bundle folder.")
    parser.add_argument("--trained-dir", required=True)
    parser.add_argument("--backend-bundle-dir", required=True)
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {src.name}")
    elif dst.exists():
        # Replacing a multimodal bundle with an image-only model must not retain
        # the previous model's schema, preprocessor or evaluation results.
        dst.unlink()
        print(f"Removed stale {dst.name}")


def main():
    args = parse_args()
    trained_dir = Path(args.trained_dir)
    bundle_dir = Path(args.backend_bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_path = trained_dir / "best_model.keras"
    if not model_path.exists():
        model_path = trained_dir / "final_model.keras"
    if not model_path.exists():
        raise FileNotFoundError("No best_model.keras or final_model.keras found in trained-dir.")

    shutil.copy2(model_path, bundle_dir / "model.keras")
    for name in [
        "class_names.json",
        "metadata_schema.json",
        "metadata_preprocessor.joblib",
        "training_config.json",
        "metrics_test.json",
        "dataset_info.json",
    ]:
        copy_if_exists(trained_dir / name, bundle_dir / name)

    print(f"Bundle exported to {bundle_dir.resolve()}")


if __name__ == "__main__":
    main()
