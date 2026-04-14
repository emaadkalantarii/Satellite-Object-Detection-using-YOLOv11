"""
yolo_stream_m_35.py
--------------------
Data preparation and YOLOv11-Medium fine-tuning script for
satellite object detection.

Steps:
    1. Restructures image directories into YOLO-compatible layout.
    2. Converts bounding box annotations from CSV into normalised
       YOLO label format (.txt files).
    3. Applies offline data augmentation to the training set using
       domain-specific transforms (sensor noise, blur, lighting
       variation, orientation flips) via the augmentation module.
    4. Fine-tunes a YOLOv11-Medium model on the prepared dataset.
"""

import os
import csv
from ultralytics import YOLO
from augmentation import augment_training_data


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_CSV  = "dataset/labels/train.csv"
VAL_CSV    = "dataset/labels/val.csv"
YAML_PATH  = "dataset/data.yaml"
TRAIN_DIR  = "dataset/train"
VAL_DIR    = "dataset/val"
IMAGE_SIZE = 1024  # All images in the dataset are 1024x1024


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------
CLASS_NAME_TO_ID = {
    "smart_1":                 0,
    "cheops":                  1,
    "lisa_pathfinder":         2,
    "debris":                  3,
    "proba_3_ocs":             4,
    "soho":                    5,
    "earth_observation_sat_1": 6,
    "proba_2":                 7,
    "xmm_newton":              8,
    "double_star":             9,
    "proba_3_csc":             10,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def move_images(directory: str) -> None:
    """
    Restructures a dataset directory into YOLO-compatible layout.
    Creates 'images/' and 'labels/' subdirectories and moves all
    image files into 'images/'.

    Args:
        directory: Path to the dataset split directory (e.g. 'dataset/train').
    """
    for subdir in ["images", "labels"]:
        os.makedirs(os.path.join(directory, subdir), exist_ok=True)

    try:
        for filename in os.listdir(directory):
            if filename in ("images", "labels"):
                continue
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, "images", filename)
            os.rename(src, dst)
            print(f"Moved: {src} → {dst}")
    except FileNotFoundError:
        print(f"Directory not found or already processed: {directory}")


def convert_csv_to_yolo_labels(csv_path: str, labels_dir: str) -> None:
    """
    Reads bounding box annotations from a CSV file and writes
    normalised YOLO-format label files (.txt) for each image.

    CSV format expected:  filename, class_name, [x1, y1, x2, y2]
    YOLO format output:   <class_id> <x_center> <y_center> <width> <height>

    All coordinates are normalised by the fixed image dimension (IMAGE_SIZE).

    Args:
        csv_path:   Path to the annotation CSV file.
        labels_dir: Directory where label .txt files will be written.
    """
    with open(csv_path, newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # Skip header row
            if "bbox" in row:
                continue

            img_filename, img_class, raw_coords = row[0], row[1], row[2]

            # Parse bounding box coordinates
            coords = raw_coords.replace("[", "").replace("]", "").split(", ")
            x1, y1, x2, y2 = (
                int(coords[0]), int(coords[1]),
                int(coords[2]), int(coords[3])
            )

            # Convert to normalised YOLO format
            x_center = round(((x1 + x2) / 2) / IMAGE_SIZE, 6)
            y_center = round(((y1 + y2) / 2) / IMAGE_SIZE, 6)
            width    = round(abs(x2 - x1) / IMAGE_SIZE, 6)
            height   = round(abs(y2 - y1) / IMAGE_SIZE, 6)

            class_id  = CLASS_NAME_TO_ID[img_class]
            label_str = f"{class_id} {x_center} {y_center} {width} {height}"

            # Write label file
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, "w") as f:
                f.write(label_str)

            print(f"Label written: {label_path}  →  {label_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1: Restructure image directories
    print("\n--- Step 1: Restructuring directories ---")
    move_images(TRAIN_DIR)
    move_images(VAL_DIR)

    # Step 2: Convert CSV annotations to YOLO label files
    print("\n--- Step 2: Converting training labels ---")
    convert_csv_to_yolo_labels(TRAIN_CSV, os.path.join(TRAIN_DIR, "labels"))

    print("\n--- Step 2: Converting validation labels ---")
    convert_csv_to_yolo_labels(VAL_CSV, os.path.join(VAL_DIR, "labels"))

    # Step 3: Offline data augmentation (training set only)
    # Augmentation is applied offline before training so that each
    # augmented image also has a correctly transformed YOLO label.
    # Validation data is intentionally left unchanged.
    # YOLO's built-in augment flag is therefore set to False in Step 4
    # to avoid double-augmenting the training images.
    print("\n--- Step 3: Applying offline data augmentation ---")
    augment_training_data(
        images_dir              = os.path.join(TRAIN_DIR, "images"),
        labels_dir              = os.path.join(TRAIN_DIR, "labels"),
        aug_images_dir          = os.path.join(TRAIN_DIR, "images"),
        aug_labels_dir          = os.path.join(TRAIN_DIR, "labels"),
        augmentations_per_image = 3,
    )

    # Step 4: Fine-tune YOLOv11-Medium
    # augment=False because offline augmentation was performed in Step 3.
    print("\n--- Step 4: Starting model training ---")
    model = YOLO("yolo11m.pt")
    model.train(
        data         = YAML_PATH,
        epochs       = 35,
        imgsz        = IMAGE_SIZE,
        batch        = 8,
        optimizer    = "auto",
        lr0          = 0.01,
        weight_decay = 0.0005,
        patience     = 100,
        augment      = False,
        device       = 0,
        workers      = 4,
        save_period  = 1,
        project      = "runs/model_m",
        name         = "model_m_35",
        verbose      = True,
    )


if __name__ == "__main__":
    main()
