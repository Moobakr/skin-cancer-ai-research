from pathlib import Path

DEFAULT_IMAGE_SIZE = 300
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42

# Only information collected by the app before a diagnosis is made.
SCREENING_METADATA_COLUMNS = ("age", "gender", "region", "smoke", "drink")

TARGET_CANDIDATES = ["diagnostic", "diagnosis", "label", "class", "target"]
PATIENT_ID_CANDIDATES = ["patient_id", "patient", "pat_id", "id_patient"]
IMAGE_COL_CANDIDATES = [
    "img_id", "image_id", "image", "filename", "file_name", "filepath", "file", "photo"
]

ID_LIKE_COLUMNS = {
    "patient_id", "patient", "pat_id", "id_patient", "lesion_id", "lesion", "img_id",
    "image_id", "image", "filename", "file_name", "filepath", "file", "photo"
}

RISK_MAP = {
    "MEL": "High risk",
    "BCC": "High risk",
    "SCC": "High risk",
    "BOD": "High risk",
    "ACK": "Medium risk",
    "AK": "Medium risk",
    "SEK": "Lower risk",
    "NEV": "Lower risk",
}

DISCLAIMER = (
    "This AI result is for educational screening support only and is not a medical diagnosis. "
    "Please consult a dermatologist for professional evaluation."
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
