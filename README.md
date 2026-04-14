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
├── yolo_stream_m_35.py     # Data preparation + augmentation + model training
├── augmentation.py         # Offline data augmentation pipeline (Albumentations)
├── Submissioncode.py       # Inference + output generation script
├── train_yolo_m.sh         # SLURM batch job script for HPC training
├── data.yaml               # YOLO dataset configuration (paths + class names)
├── requirements.txt        # Python dependencies
└── dataset/                # (Not included — see Data Setup below)
    ├── labels/
    │   ├── train.csv
    │   └── val.csv
    ├── train/
    ├── val/
    └── test/
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv11 model training and inference |
| `opencv-python` | Image reading and colour-space conversion |
| `albumentations` | Offline data augmentation pipeline |
| `torch` | PyTorch backend used by Ultralytics |

> `os`, `csv`, `random`, and `pathlib` are Python standard library modules — no installation needed.

---

## Data Setup

### Dataset Source

This project uses the **SPARK 2022 Challenge** dataset (Stream 1), provided by the University of Luxembourg's Interdisciplinary Centre for Security, Reliability and Trust (SnT). The dataset contains labelled satellite imagery across 11 object categories including real ESA spacecraft and space debris.

> Dataset access: [SPARK 2022 — Stream 1](https://gitlab.com/uniluxembourg/snt/cvi2/space/spark-challenge/spark2022-utils/-/tree/main/stream-1)

The dataset is not included in this repository due to size. To reproduce the setup:

1. Download Stream 1 from the link above and place training and validation images under `dataset/train/` and `dataset/val/` respectively.
2. Place annotation files `train.csv` and `val.csv` under `dataset/labels/`.
3. Each CSV row follows the format: `filename, class_name, [x1, y1, x2, y2]`
4. All images are 1024×1024 pixels.

---

## Pipeline Overview

`yolo_stream_m_35.py` runs the full pipeline in four sequential steps:

### Step 1 — Directory Restructuring

Moves images into the `images/` subdirectory and creates the `labels/` subdirectory as required by YOLO format.

### Step 2 — Label Conversion

Reads bounding box annotations from the CSV files and converts them to normalised YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised by dividing by 1024 (the fixed image dimension).

### Step 3 — Offline Data Augmentation

Before training, `augmentation.py` applies a domain-specific augmentation pipeline to the training set using the [Albumentations](https://albumentations.ai/) library. Each training image is augmented **3 times**, generating additional labelled samples with correctly transformed bounding boxes.

The pipeline includes the following transforms, each motivated by real satellite imaging physics:

| Transform | Simulation Target |
|-----------|------------------|
| `HorizontalFlip`, `VerticalFlip` | Arbitrary orbital viewing angles and passes |
| `RandomRotate90` | Satellite orientation variability |
| `RandomBrightnessContrast`, `CLAHE` | Sunlight angle variation across the orbit |
| `GaussNoise` | CCD sensor noise in low-light / deep-space conditions |
| `GaussianBlur`, `MotionBlur` | Optical defocus and micro-vibration from attitude control thrusters |
| `CoarseDropout` | Sensor dead pixels and telemetry data dropout |

Bounding boxes are transformed alongside each image using Albumentations' `BboxParams` with `format="yolo"`, ensuring label accuracy is preserved after every transform. Validation data is intentionally left unaugmented.

> **Note:** Because augmentation is handled offline in this step, YOLO's built-in `augment` flag is set to `False` during training to avoid double-augmenting the training images.

### Step 4 — Model Training

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
| Built-in augmentation | False (handled offline in Step 3) |

Training outputs (weights, logs, metrics) are saved to `runs/model_m/`.

---

## Running on a Local Machine (without HPC)

If you do not have access to an HPC cluster, you can run the full pipeline on your own machine. The following guidance covers the most common setups.

### Hardware Requirements

| Setup | Recommendation |
|-------|---------------|
| **GPU (recommended)** | NVIDIA GPU with ≥ 8 GB VRAM (e.g. RTX 3070, RTX 4080) |
| **CPU only** | Possible but very slow — expect 10–30× longer training time |
| **RAM** | ≥ 16 GB recommended due to the 1024×1024 image size |
| **Disk** | ≥ 10 GB free for dataset, augmented images, and model outputs |

> Training 35 epochs at 1024px on a mid-range GPU (e.g. RTX 3070) typically takes **3–6 hours**. On CPU it may take **24+ hours**.

### Step-by-Step Local Setup

**1. Clone the repository**

```bash
git clone https://github.com/emaadkalantarii/Satellite-Object-Detection-using-YOLOv11.git
cd Satellite-Object-Detection-using-YOLOv11
```

**2. Create and activate a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download and place the dataset**

Download Stream 1 of the SPARK 2022 dataset from the link in the [Data Setup](#data-setup) section, then organise it as follows:

```
dataset/
├── labels/
│   ├── train.csv
│   └── val.csv
├── train/        ← place training images here
├── val/          ← place validation images here
└── test/         ← place test images here (for inference only)
```

**5. Verify `data.yaml` paths**

The provided `data.yaml` already uses relative paths (`dataset/train/images`, `dataset/val/images`) which work out of the box when running from the root of the repository. If you place the dataset elsewhere, update those two paths accordingly.

**6. Run the full training pipeline**

```bash
python yolo_stream_m_35.py
```

This single command runs all four steps automatically: directory restructuring → label conversion → offline augmentation → model training. Progress is printed to the terminal at each step.

**Adjusting for limited VRAM:** If you run out of GPU memory, reduce the batch size or image size in `yolo_stream_m_35.py`:

```python
batch  = 4    # reduce from 8 if GPU VRAM < 8 GB
imgsz  = 640  # reduce from 1024 for faster training on smaller GPUs
```

**Training on CPU instead of GPU:**

```python
device = "cpu"   # change from device = 0
```

**7. Run inference on test images**

Once training completes, the best model weights are saved automatically to `runs/model_m/model_m_35/weights/best.pt`. Then run:

```bash
python Submissioncode.py
```

Results are saved to `dataset/submission.csv` with columns: `filename`, `class`, `bbox`.

---

## Training on HPC with SLURM

Training was originally executed on the **University of Luxembourg HPC Iris cluster** using a dedicated GPU node, managed via **SLURM** job scheduling.

Submit the batch script as follows:

```bash
sbatch train_yolo_m.sh
```

Key SLURM configuration in `train_yolo_m.sh`:

| Parameter | Value |
|-----------|-------|
| GPUs | 1 per task |
| CPUs | 7 per GPU |
| Max walltime | 2 days |
| Conda environment | `yolov11` |

> If you are using a different HPC cluster, update the `#SBATCH` directives and conda environment name in `train_yolo_m.sh` to match your cluster's configuration.

---

## Results & Evaluation

The final trained model was evaluated against a held-out labelled test set, achieving an overall classification accuracy of approximately **86%**.

### Training Metrics

The plots below show training and validation loss curves alongside key detection metrics across the first 10 checkpoint epochs:

<img width="2400" height="1200" alt="results_m" src="https://github.com/user-attachments/assets/d48038cc-f5de-4319-a267-51f23c743689" />

| Metric | Trend |
|--------|-------|
| `train/box_loss` | Steadily decreasing — model improves bounding box localisation |
| `train/cls_loss` | Strongly decreasing — class discrimination improves rapidly |
| `train/dfl_loss` | Decreasing — distribution focal loss improves box regression |
| `val/box_loss` | Decreasing and closely tracks training loss — good generalisation |
| `val/cls_loss` | Decreasing without divergence from training — no overfitting |
| `metrics/precision(B)` | Rising toward ~0.48 — improving positive prediction reliability |
| `metrics/recall(B)` | Rising toward ~0.83 — model increasingly detects true objects |
| `metrics/mAP50(B)` | Rising toward ~0.50 — solid detection performance at IoU 0.50 |
| `metrics/mAP50-95(B)` | Rising toward ~0.50 — strong localisation quality across IoU thresholds |

The close alignment between training and validation curves across all loss metrics confirms that the model generalises well and is not overfitting despite the domain-specific and relatively small dataset.

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

- Model weights (`.pt` files) are excluded via `.gitignore` due to file size; they are generated locally during training.
- The `dataset/` directory is excluded; annotations and images must be sourced from the SPARK 2022 link above.
- The results metrics plot shows checkpointed metrics across the first 10 epochs of the 35-epoch training run.
- The `train_yolo_m.sh` SLURM script is included for HPC reproducibility but is not needed for local runs.
