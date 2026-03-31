# augmentation.py
# Offline data augmentation pipeline for satellite object detection.
# Applies domain-specific augmentations to simulate real-world orbital imaging
# conditions (sensor noise, motion blur, lighting variation, orientation changes).
# Generates augmented training images alongside correctly transformed YOLO labels.

import os
import cv2
import random
import albumentations as A
from pathlib import Path


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TRAIN_IMAGES_DIR = "dataset/train/images"   # Input training images
TRAIN_LABELS_DIR = "dataset/train/labels"   # Corresponding YOLO .txt labels
AUG_IMAGES_DIR   = "dataset/train/images"   # Augmented images saved in-place
AUG_LABELS_DIR   = "dataset/train/labels"   # Augmented labels saved in-place

AUGMENTATIONS_PER_IMAGE = 3   # How many augmented versions to create per image
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# Augmentation Pipeline
# ─────────────────────────────────────────────
# Each transform is chosen to simulate a specific real-world satellite imaging
# condition. Probabilities (p) are tuned to keep augmentations realistic —
# not every image should receive every transform.

def build_augmentation_pipeline() -> A.Compose:
    """
    Builds and returns the Albumentations augmentation pipeline.
    Bounding boxes are passed through in YOLO format and transformed
    correctly alongside the image.
    """
    return A.Compose(
        [
            # ── Orientation transforms ─────────────────────────────────────
            # Satellites are imaged from arbitrary angles; flips simulate
            # different orbital passes and viewing geometries.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),

            # ── Lighting / contrast variation ──────────────────────────────
            # Sunlight angle changes dramatically across an orbit, causing
            # strong brightness and contrast shifts in imagery.
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6
            ),
            A.CLAHE(
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),

            # ── Sensor and atmospheric noise ───────────────────────────────
            # CCD sensors on satellites introduce Gaussian noise, especially
            # in low-light or deep-space imaging conditions.
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                mean=0,
                p=0.4
            ),

            # ── Blur effects ───────────────────────────────────────────────
            # Gaussian blur simulates optical defocus or atmospheric effects.
            # Motion blur simulates micro-vibrations from satellite attitude
            # control thrusters or orbital velocity.
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.35
            ),

            # ── Coarse dropout / occlusion ─────────────────────────────────
            # Simulates partial occlusion by debris, sensor artifacts,
            # or data transmission dropouts in satellite telemetry.
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(10, 40),
                hole_width_range=(10, 40),
                fill=0,
                p=0.2
            ),
        ],
        # Tell Albumentations that bounding boxes are in YOLO format
        # (x_center, y_center, width, height — all normalized 0–1).
        # min_visibility=0.3 drops a bbox only if >70% of it is cropped away.
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            clip=True
        )
    )


# ─────────────────────────────────────────────
# Label I/O helpers
# ─────────────────────────────────────────────

def read_yolo_label(label_path: str):
    """
    Reads a YOLO-format .txt label file.
    Returns:
        bboxes       : list of [x_center, y_center, width, height] (normalized)
        class_labels : list of int class IDs
    """
    bboxes, class_labels = [], []
    if not os.path.exists(label_path):
        return bboxes, class_labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            class_labels.append(cls)
            bboxes.append(coords)

    return bboxes, class_labels


def write_yolo_label(label_path: str, bboxes: list, class_labels: list):
    """
    Writes bounding boxes and class labels to a YOLO-format .txt file.
    """
    with open(label_path, "w") as f:
        for cls, bbox in zip(class_labels, bboxes):
            x_c, y_c, w, h = bbox
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


# ─────────────────────────────────────────────
# Main augmentation loop
# ─────────────────────────────────────────────

def augment_training_data(
    images_dir: str = TRAIN_IMAGES_DIR,
    labels_dir: str = TRAIN_LABELS_DIR,
    aug_images_dir: str = AUG_IMAGES_DIR,
    aug_labels_dir: str = AUG_LABELS_DIR,
    augmentations_per_image: int = AUGMENTATIONS_PER_IMAGE,
):
    """
    Iterates over all training images, applies the augmentation pipeline
    N times per image, and saves the results with matching YOLO labels.

    Args:
        images_dir            : Directory containing original training images.
        labels_dir            : Directory containing original YOLO .txt labels.
        aug_images_dir        : Directory to save augmented images.
        aug_labels_dir        : Directory to save augmented labels.
        augmentations_per_image: Number of augmented copies to create per image.
    """
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)

    pipeline = build_augmentation_pipeline()

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n{'='*60}")
    print(f"Satellite Imagery Augmentation Pipeline")
    print(f"{'='*60}")
    print(f"  Source images : {len(image_files)}")
    print(f"  Augmentations : {augmentations_per_image} per image")
    print(f"  Total new samples to generate: "
          f"{len(image_files) * augmentations_per_image}")
    print(f"{'='*60}\n")

    total_generated = 0
    total_skipped   = 0

    for img_file in image_files:
        img_stem  = Path(img_file).stem
        img_path  = os.path.join(images_dir, img_file)
        lbl_path  = os.path.join(labels_dir, img_stem + ".txt")

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [WARN] Could not read image: {img_file}. Skipping.")
            total_skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read corresponding YOLO label
        bboxes, class_labels = read_yolo_label(lbl_path)

        # Skip images with no valid annotations
        if not bboxes:
            print(f"  [WARN] No valid labels for: {img_file}. Skipping.")
            total_skipped += 1
            continue

        # Generate N augmented versions
        for aug_idx in range(augmentations_per_image):
            try:
                augmented = pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                aug_image  = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]

                # Skip if all bboxes were removed (object became invisible)
                if not aug_bboxes:
                    continue

                # Build output filenames:  originalname_aug0.jpg / .txt
                out_name      = f"{img_stem}_aug{aug_idx}"
                out_img_path  = os.path.join(aug_images_dir, out_name + ".jpg")
                out_lbl_path  = os.path.join(aug_labels_dir, out_name + ".txt")

                # Save augmented image (convert back to BGR for OpenCV)
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_img_path, aug_image_bgr)

                # Save augmented YOLO label
                write_yolo_label(out_lbl_path, aug_bboxes, aug_labels)

                total_generated += 1

            except Exception as e:
                print(f"  [ERROR] Augmentation failed for {img_file} "
                      f"(aug {aug_idx}): {e}")

    print(f"\nAugmentation complete.")
    print(f"  Images generated : {total_generated}")
    print(f"  Images skipped   : {total_skipped}")
    print(f"  New dataset size : original + {total_generated} samples\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    augment_training_data()
