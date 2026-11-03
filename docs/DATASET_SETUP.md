# Dataset setup

Dataset: **Skin Cancer (PAD-UFES-20)**.

The project expects a folder containing:

```text
data/pad_ufes_20/
├── metadata.csv
└── images/
    └── image files
```

## Option A: Manual Kaggle download

1. Open the Kaggle dataset page.
2. Download and unzip the dataset.
3. Move the extracted files into `data/pad_ufes_20/`.

## Option B: Kaggle API

Install Kaggle API:

```bash
pip install kaggle
```

Place `kaggle.json` in:

- Linux/macOS: `~/.kaggle/kaggle.json`
- Windows: `C:\Users\<YOUR_USER>\.kaggle\kaggle.json`

Then run:

```bash
mkdir -p data/pad_ufes_20
kaggle datasets download -d mahdavi1202/skin-cancer -p data/pad_ufes_20 --unzip
```

## Common metadata columns

The training code tries to detect common names automatically, including:

- target/label: `diagnostic`, `diagnosis`, `label`, `class`
- patient id: `patient_id`, `patient`, `pat_id`
- image id/file: `img_id`, `image_id`, `filename`, `file_name`, `image`, `photo`

If your CSV names are different, pass explicit arguments:

```bash
python -m src.train_image_model \
  --dataset-dir ../data/pad_ufes_20 \
  --target-col diagnostic \
  --patient-col patient_id \
  --image-col img_id
```
