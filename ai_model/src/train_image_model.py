from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .config import DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_SEED, ensure_dir
from .data import load_dataset_dataframe, make_patient_level_split, save_split_files
from .models import build_custom_cnn, build_efficientnet_classifier, fine_tune_efficientnet
from .preprocessing import make_image_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train image-only skin lesion classifier.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--image-col", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=["efficientnet", "custom"], default="efficientnet")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


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

    train_df, val_df, test_df = make_patient_level_split(df, target_col, patient_col, seed=args.seed)
    save_split_files(output_dir, train_df, val_df, test_df, info)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[target_col])
    class_names = label_encoder.classes_.tolist()
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    y_train = label_encoder.transform(train_df[target_col])
    y_val = label_encoder.transform(val_df[target_col])

    train_ds = make_image_dataset(
        train_df["image_path"].values, y_train, args.image_size, args.batch_size, shuffle=True, augment=True, seed=args.seed
    )
    val_ds = make_image_dataset(
        val_df["image_path"].values, y_val, args.image_size, args.batch_size, shuffle=False, augment=False
    )

    if args.model == "custom":
        model = build_custom_cnn(args.image_size, len(class_names), learning_rate=1e-3)
    else:
        model = build_efficientnet_classifier(args.image_size, len(class_names), learning_rate=1e-3, base_trainable=False)

    classes = np.arange(len(class_names))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(i): float(w) for i, w in zip(classes, class_weights)}

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / "best_model.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv", append=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    if args.model == "efficientnet" and args.fine_tune_epochs > 0:
        model = fine_tune_efficientnet(model, learning_rate=1e-5, unfreeze_last=25)
        warmup_epochs = len(history.epoch)
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=warmup_epochs,
            epochs=warmup_epochs + args.fine_tune_epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=2,
        )

    model.save(output_dir / "final_model.keras")
    training_config = vars(args) | {"dataset_info": info, "class_weight": class_weight_dict}
    (output_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    print(f"Training complete. Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
