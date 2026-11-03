from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .config import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_SEED, ensure_dir
from .data import load_dataset_dataframe, make_patient_level_split, metadata_feature_columns, save_split_files
from .models import build_multimodal_classifier, fine_tune_efficientnet
from .preprocessing import make_multimodal_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train image + metadata multimodal classifier.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def build_metadata_preprocessor(train_df, feature_cols):
    X = train_df[feature_cols].copy()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    df, info = load_dataset_dataframe(
        args.dataset_dir,
        metadata_csv=args.metadata_csv,
        target_col=args.target_col,
        patient_col=args.patient_col,
        image_col=args.image_col,
    )
    target_col = info["target_col"]
    patient_col = info["patient_col"]
    image_col = info["image_col"]

    feature_cols = metadata_feature_columns(df, target_col, extra_exclude=[image_col, patient_col])
    if not feature_cols:
        raise ValueError("No metadata feature columns found. Use image-only training instead.")

    train_df, val_df, test_df = make_patient_level_split(df, target_col, patient_col, seed=args.seed)
    # Match the image-only baseline: class weighting without extra oversampling.
    save_split_files(output_dir, train_df, val_df, test_df, info)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[target_col])
    class_names = label_encoder.classes_.tolist()
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    preprocessor, numeric_cols, categorical_cols = build_metadata_preprocessor(train_df, feature_cols)
    X_train_meta = preprocessor.fit_transform(train_df[feature_cols])
    X_val_meta = preprocessor.transform(val_df[feature_cols])
    joblib.dump(preprocessor, output_dir / "metadata_preprocessor.joblib")

    metadata_schema = {
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "metadata_dim": int(X_train_meta.shape[1]),
    }
    (output_dir / "metadata_schema.json").write_text(json.dumps(metadata_schema, indent=2), encoding="utf-8")

    y_train = label_encoder.transform(train_df[target_col])
    y_val = label_encoder.transform(val_df[target_col])

    train_ds = make_multimodal_dataset(
        train_df["image_path"].values,
        X_train_meta,
        y_train,
        args.image_size,
        args.batch_size,
        shuffle=True,
        augment=True,
        seed=args.seed,
    )
    val_ds = make_multimodal_dataset(
        val_df["image_path"].values,
        X_val_meta,
        y_val,
        args.image_size,
        args.batch_size,
        shuffle=False,
        augment=False,
    )

    model = build_multimodal_classifier(
        image_size=args.image_size,
        metadata_dim=X_train_meta.shape[1],
        num_classes=len(class_names),
        learning_rate=1e-3,
        base_trainable=False,
    )

    classes = np.arange(len(class_names))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(i): float(w) for i, w in zip(classes, class_weights)}

    previous_best = None
    log_path = output_dir / "training_log.csv"
    if log_path.exists() and log_path.stat().st_size:
        previous_log = pd.read_csv(log_path)
        if len(previous_log):
            previous_best = float(previous_log.val_loss.min())
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / "best_model.keras", monitor="val_loss", save_best_only=True, initial_value_threshold=previous_best),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv", append=True),
    ]

    warmup_state_path = output_dir / "warmup_state.json"
    warmup_model_path = output_dir / "warmup_model.keras"
    if warmup_state_path.exists():
        warmup_epochs = json.loads(warmup_state_path.read_text())["completed_epochs"]
    else:
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks + [tf.keras.callbacks.BackupAndRestore(output_dir / "warmup_backup")],
            verbose=2,
        )
        warmup_epochs = max(history.epoch) + 1
        # Keep the frozen-stage best separately from the global checkpoint.
        import shutil
        shutil.copy2(output_dir / "best_model.keras", warmup_model_path)
        warmup_state_path.write_text(json.dumps({"completed_epochs": warmup_epochs}), encoding="utf-8")

    # Reuse the existing model to avoid keeping a second backbone in RAM.
    model.load_weights(warmup_model_path)

    if args.fine_tune_epochs > 0:
        model = fine_tune_efficientnet(model, learning_rate=1e-5, unfreeze_last=25)
        # Keras BackupAndRestore leaves this private resume marker on the model.
        # A fine-stage backup, when present, sets its own epoch in on_train_begin.
        model._initial_epoch = None
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=warmup_epochs,
            epochs=warmup_epochs + args.fine_tune_epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks + [tf.keras.callbacks.BackupAndRestore(output_dir / "fine_tuning_backup")],
            verbose=2,
        )

    model.save(output_dir / "final_model.keras")
    training_config = vars(args) | {"dataset_info": info, "class_weight": class_weight_dict, "metadata_schema": metadata_schema}
    (output_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    print(f"Multimodal training complete. Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
