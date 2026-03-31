# Satellite Object Detection using YOLOv11

A computer vision system for detecting and classifying 10 real ESA spacecraft and space debris from 1024×1024 satellite imagery, built by fine-tuning a YOLOv11-Medium model on a custom annotated dataset.

---

## Classes

The model detects 11 object categories:

| ID | Class |
|----|-------|
| 0  | smart_1 |
| 1  | cheops |
| 2  | lisa_pathfinder |
| 3  | debris |
| 4  | proba_3_ocs |
| 5  | soho |
| 6  | earth_observation_sat_1 |
| 7  | proba_2 |
| 8  | xmm_newton |
| 9  | double_star |
| 10 | proba_3_csc |

---

## Repository Structure

```
├── yolo_stream_m_35.py     # Data preparation + model training script
├── Submissioncode.py       # Inference + output generation script
├── train_yolo_m.sh         # SLURM batch job script for HPC training
├── data.yaml               # YOLO dataset configuration (paths + class names)
├── requirements.txt        # Python dependencies
└── dataset/                # (Not included - see Data Setup below)
    ├── labels/
    │   ├── train.csv
    │   └── val.csv
    ├── train/
    ├── val/
    └── test/
```

---

## Dependencies

Only the following libraries are required:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv11 model training and inference |
| `opencv-python` | Image reading and colour-space conversion |
| `torch` | PyTorch backend for model evaluation |

> `os` and `csv` are Python standard library modules — no installation needed.

---

## Data Setup

The dataset is not included in this repository due to size. To reproduce the setup:

1. Place training and validation images under `dataset/train/` and `dataset/val/` respectively.
2. Place annotation files `train.csv` and `val.csv` under `dataset/labels/`.
3. Each CSV row should follow the format: `filename, class_name, [x1, y1, x2, y2]`
4. All images are expected to be **1024×1024 pixels**.

---

## Data Preparation

`yolo_stream_m_35.py` handles two preparation steps automatically:

**1. Directory restructuring** — Moves images into the `images/` subdirectory and creates the `labels/` subdirectory, as required by the YOLO format.

**2. Label conversion** — Reads bounding box annotations from the CSV files and converts them to normalised YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised by dividing by 1024 (the fixed image dimension).

---

## Model Training

The YOLOv11-Medium model (`yolo11m.pt`) is fine-tuned using the following configuration:

| Parameter | Value |
|-----------|-------|
| Epochs | 35 |
| Image size | 1024 × 1024 |
| Batch size | 8 |
| Optimizer | Auto |
| Initial LR | 0.01 |
| Weight decay | 0.0005 |
| Early stopping patience | 100 |

Training outputs (weights, logs, metrics) are saved to `runs/model_m/`.

---

## Training Environment

Training was executed on the **University of Luxembourg HPC Iris cluster** using a dedicated GPU node, managed via **SLURM** job scheduling.

The batch script `train_yolo_m.sh` was submitted as follows:

```bash
sbatch train_yolo_m.sh
```

Key SLURM configuration:
- 1 GPU per task
- 7 CPUs per GPU
- Max walltime: 2 days
- Conda environment: `yolov11`

---

## Inference

`Submissioncode.py` runs inference on all images in `dataset/test/` and produces a CSV output with the following columns:

| Column | Description |
|--------|-------------|
| `filename` | Image filename |
| `class` | Predicted class name |
| `bbox` | Bounding box as `x1,y1,x2,y2` |

To run inference:

```bash
python Submissioncode.py
```

Output is saved to `dataset/submission.csv`.

---

## Notes

- Model weights (`.pt` files) are excluded from the repository via `.gitignore` due to file size.
- The `dataset/` directory is also excluded; annotations and images must be sourced separately.
