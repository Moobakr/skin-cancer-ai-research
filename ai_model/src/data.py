from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import (
    IMAGE_COL_CANDIDATES,
    PATIENT_ID_CANDIDATES,
    TARGET_CANDIDATES,
    ID_LIKE_COLUMNS,
    DEFAULT_SEED,
    SCREENING_METADATA_COLUMNS,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _first_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def find_metadata_csv(dataset_dir: str | Path) -> Path:
    dataset_dir = Path(dataset_dir)
    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found under {dataset_dir}")

    # Prefer names containing metadata.
    for csv_file in csv_files:
        if "metadata" in csv_file.name.lower():
            return csv_file
    return csv_files[0]


def build_image_index(dataset_dir: str | Path) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)
    index: dict[str, Path] = {}
    for path in dataset_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name] = path
            index[path.stem] = path
    return index


def load_dataset_dataframe(
    dataset_dir: str | Path,
    metadata_csv: str | Path | None = None,
    target_col: str | None = None,
    patient_col: str | None = None,
    image_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load metadata and attach an absolute image_path column.

    The function auto-detects common PAD-UFES-20 column names, but explicit column
    arguments should be used if your downloaded CSV has a different schema.
    """
    dataset_dir = Path(dataset_dir)
    metadata_path = Path(metadata_csv) if metadata_csv else find_metadata_csv(dataset_dir)
    df = pd.read_csv(metadata_path)
    df.columns = [str(c).strip() for c in df.columns]

    target_col = target_col or _first_existing_column(df.columns, TARGET_CANDIDATES)
    patient_col = patient_col or _first_existing_column(df.columns, PATIENT_ID_CANDIDATES)
    image_col = image_col or _first_existing_column(df.columns, IMAGE_COL_CANDIDATES)

    if target_col is None:
        raise ValueError(
            "Could not detect target column. Pass --target-col explicitly. "
            f"Available columns: {list(df.columns)}"
        )
    if image_col is None:
        raise ValueError(
            "Could not detect image filename column. Pass --image-col explicitly. "
            f"Available columns: {list(df.columns)}"
        )

    image_index = build_image_index(dataset_dir)

    def resolve_image(value):
        if pd.isna(value):
            return None
        value = str(value).strip()
        path = Path(value)
        if path.exists():
            return str(path.resolve())
        if value in image_index:
            return str(image_index[value].resolve())
        if path.stem in image_index:
            return str(image_index[path.stem].resolve())
        # Try common extension additions.
        for suffix in IMAGE_SUFFIXES:
            key = value + suffix
            if key in image_index:
                return str(image_index[key].resolve())
        return None

    df["image_path"] = df[image_col].apply(resolve_image)
    df = df.dropna(subset=["image_path", target_col]).copy()
    df[target_col] = df[target_col].astype(str).str.strip().str.upper()

    if patient_col is None:
        # Fallback: each image is treated as its own group.
        patient_col = "_synthetic_patient_id"
        df[patient_col] = df[image_col].astype(str)

    info = {
        "metadata_csv": str(metadata_path),
        "target_col": target_col,
        "patient_col": patient_col,
        "image_col": image_col,
        "num_rows": int(len(df)),
        "classes": sorted(df[target_col].unique().tolist()),
    }
    return df, info


def make_patient_level_split(
    df: pd.DataFrame,
    target_col: str,
    patient_col: str,
    seed: int = DEFAULT_SEED,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by patient/group to reduce leakage."""
    groups = df[patient_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(df, df[target_col], groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    val_relative_size = val_size / (1.0 - test_size)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=seed + 1)
    groups_train_val = train_val_df[patient_col].astype(str)
    train_idx, val_idx = next(val_splitter.split(train_val_df, train_val_df[target_col], groups_train_val))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def oversample_training_set(df: pd.DataFrame, target_col: str, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Oversample minority classes in the training set up to the median class count.

    Capped at 3x per class to avoid the model memorising augmented duplicates.
    Never touches the validation or test splits.
    """
    class_counts = df[target_col].value_counts()
    target_count = int(class_counts.median())
    parts = []
    for cls, count in class_counts.items():
        cls_df = df[df[target_col] == cls]
        if count < target_count:
            multiplier = min(int(target_count / count), 3)
            cls_df = pd.concat([cls_df] * multiplier, ignore_index=True)
        parts.append(cls_df)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def metadata_feature_columns(df: pd.DataFrame, target_col: str, extra_exclude: Iterable[str] = ()) -> list[str]:
    exclude = set(ID_LIKE_COLUMNS) | {target_col, "image_path"} | set(extra_exclude)
    # An allowlist prevents biopsy status and other diagnostic-process fields
    # from silently becoming model inputs when the dataset schema changes.
    return [c for c in SCREENING_METADATA_COLUMNS if c in df.columns and c not in exclude]


def save_split_files(output_dir: str | Path, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, info: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
