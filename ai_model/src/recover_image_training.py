"""Restart the interrupted fine-tuning phase from its saved frozen checkpoint.

Keeps the original log, split and global best checkpoint. This is a documented
phase restart, not an exact restoration of the interrupted optimizer/RNG state.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .models import fine_tune_efficientnet
from .preprocessing import make_image_dataset


def main():
    output = Path("artifacts/image_screening")
    if (output / "training_config.json").exists():
        print("Image training already completed; keeping its artifacts.")
        return
    tf.keras.utils.set_random_seed(42)
    checkpoint = output / "best_model.keras"
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    base = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name)
    if base.trainable and not (output / "recovery_origin.json").exists():
        raise ValueError("Expected the saved frozen-backbone checkpoint for this recovery.")
    origin = output / "recovery_origin.json"
    if origin.exists():
        source_hash = json.loads(origin.read_text())["source_sha256"]
    else:
        origin.write_text(json.dumps({"source_sha256": source_hash}), encoding="utf-8")
    log = pd.read_csv(output / "training_log.csv")
    warmup_epochs = int((log.learning_rate > 1e-5).sum())
    best_loss = float(log.val_loss.min())
    recovery_log = output / "recovery_fine_tuning_log.csv"
    if recovery_log.exists() and recovery_log.stat().st_size:
        prior = pd.read_csv(recovery_log)
        if len(prior):
            best_loss = min(best_loss, float(prior.val_loss.min()))
    classes = json.loads((output / "class_names.json").read_text())
    info = json.loads((output / "dataset_info.json").read_text())
    train = pd.read_csv(output / "train_split.csv")
    val = pd.read_csv(output / "val_split.csv")
    mapping = {name: index for index, name in enumerate(classes)}
    y_train = train[info["target_col"]].map(mapping).to_numpy()
    y_val = val[info["target_col"]].map(mapping).to_numpy()
    if pd.isna(y_train).any() or pd.isna(y_val).any():
        raise ValueError("Unknown labels in saved splits.")
    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_train)
    class_weights = {index: float(value) for index, value in enumerate(weights)}
    train_ds = make_image_dataset(train.image_path.values, y_train, 300, 8, shuffle=True, augment=True, seed=42)
    val_ds = make_image_dataset(val.image_path.values, y_val, 300, 8)
    model = fine_tune_efficientnet(model, learning_rate=1e-5, unfreeze_last=25)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True, initial_value_threshold=best_loss),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", baseline=best_loss, patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(recovery_log, append=True),
        tf.keras.callbacks.BackupAndRestore(output / "recovery_backup"),
    ]
    model.fit(train_ds, validation_data=val_ds, initial_epoch=warmup_epochs,
              epochs=warmup_epochs + 8, class_weight=class_weights, callbacks=callbacks, verbose=2)
    model.save(output / "final_model.keras")
    config = {
        "dataset_dir": "../data/pad_ufes_20", "output_dir": str(output), "model": "efficientnet",
        "epochs": 25, "fine_tune_epochs": 8, "batch_size": 8, "image_size": 300, "seed": 42,
        "dataset_info": info, "class_weight": class_weights,
        "recovery": {"method": "fine-tuning phase restarted from saved frozen checkpoint",
                     "source_sha256": source_hash, "completed_warmup_epochs": warmup_epochs,
                     "interrupted_fine_tuning_epochs": len(log) - warmup_epochs,
                     "exact_optimizer_or_rng_resume": False},
    }
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Image fine-tuning recovery completed.", flush=True)


if __name__ == "__main__":
    main()
