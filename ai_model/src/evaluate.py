from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from .config import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE
from .preprocessing import make_image_dataset, make_multimodal_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a saved split.")
    parser.add_argument("--trained-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    trained_dir = Path(args.trained_dir)
    model_path = trained_dir / "best_model.keras"
    if not model_path.exists():
        model_path = trained_dir / "final_model.keras"
    # Prediction does not need the custom focal-loss closure used in training.
    model = tf.keras.models.load_model(model_path, compile=False)

    dataset_info = json.loads((trained_dir / "dataset_info.json").read_text(encoding="utf-8"))
    target_col = dataset_info["target_col"]
    df = pd.read_csv(trained_dir / f"{args.split}_split.csv")

    class_names = json.loads((trained_dir / "class_names.json").read_text(encoding="utf-8"))
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)
    y_true = label_encoder.transform(df[target_col].astype(str).str.upper())

    metadata_schema_path = trained_dir / "metadata_schema.json"
    if metadata_schema_path.exists():
        schema = json.loads(metadata_schema_path.read_text(encoding="utf-8"))
        preprocessor = joblib.load(trained_dir / "metadata_preprocessor.joblib")
        X_meta = preprocessor.transform(df[schema["feature_columns"]])
        ds = make_multimodal_dataset(
            df["image_path"].values, X_meta, y_true, args.image_size, args.batch_size, shuffle=False, augment=False
        )
    else:
        ds = make_image_dataset(df["image_path"].values, y_true, args.image_size, args.batch_size, shuffle=False, augment=False)

    proba = model.predict(ds)
    y_pred = np.argmax(proba, axis=1)

    labels = np.arange(len(class_names))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": class_names,
        "split": args.split,
        "sample_count": len(df),
        "inference": "single_view",
        "legacy_biopsy_input": metadata_schema_path.exists() and "biopsed" in schema["feature_columns"],
    }

    output_path = trained_dir / f"metrics_{args.split}.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Keep sample-level results for error analysis without another inference run.
    predictions = df[[dataset_info["image_col"], dataset_info["patient_col"]]].copy()
    predictions["true_class"] = np.asarray(class_names)[y_true]
    predictions["predicted_class"] = np.asarray(class_names)[y_pred]
    predictions["confidence"] = np.max(proba, axis=1)
    for index, name in enumerate(class_names):
        predictions[f"probability_{name}"] = proba[:, index]
    predictions.to_csv(trained_dir / f"predictions_{args.split}.csv", index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))
    print(f"Full metrics saved to {output_path}")


if __name__ == "__main__":
    main()
