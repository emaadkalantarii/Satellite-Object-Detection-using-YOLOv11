"""
augmentation.py
---------------
Offline data augmentation pipeline for satellite object detection.

Applies domain-specific image transforms to simulate real-world orbital
imaging conditions (sensor noise, motion blur, lighting variation, and
orientation changes). Each augmented image is saved alongside a correctly
transformed YOLO-format label file, ensuring bounding boxes remain
accurately aligned after every transform.

Intended to be called from yolo_stream_m_35.py before model training.
"""

import os
import cv2
import random
import albumentations as A
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUGMENTATIONS_PER_IMAGE = 3
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_augmentation_pipeline() -> A.Compose:
    """
    Builds and returns the Albumentations augmentation pipeline.

    Each transform is chosen to simulate a specific real-world satellite
    imaging condition. Bounding boxes are passed through in YOLO format
    and transformed correctly alongside each image.

    Returns:
        An Albumentations Compose pipeline with bbox support.
    """
    return A.Compose(
        [
            # Orientation transforms:
            # Satellites are imaged from arbitrary angles; flips simulate
            # different orbital passes and viewing geometries.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),

            # Lighting / contrast variation:
            # Sunlight angle changes dramatically across an orbit, causing
            # strong brightness and contrast shifts in captured imagery.
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6,
            ),
            A.CLAHE(
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                p=0.3,
            ),

            # Sensor noise:
            # CCD sensors on satellites introduce Gaussian noise, especially
            # in low-light or deep-space imaging conditions.
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                mean=0,
                p=0.4,
            ),

            # Blur effects:
            # Gaussian blur simulates optical defocus or atmospheric effects.
            # Motion blur simulates micro-vibrations from attitude control
            # thrusters or orbital velocity during image capture.
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.35,
            ),

            # Coarse dropout / occlusion:
            # Simulates partial occlusion by debris, sensor dead pixels,
            # or data dropout in satellite telemetry transmission.
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(10, 40),
                hole_width_range=(10, 40),
                fill=0,
                p=0.2,
            ),
        ],
        # Bounding boxes are provided in YOLO format:
        # (x_center, y_center, width, height) — all normalised to [0, 1].
        # min_visibility=0.3 discards a bbox only if more than 70% of it
        # has been cropped or occluded by a transform.
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            clip=True,
        ),
    )


def read_yolo_label(label_path: str):
    """
    Reads a YOLO-format .txt label file.

    Args:
        label_path: Full path to the .txt label file.

    Returns:
        bboxes:       list of [x_center, y_center, width, height] (normalised).
        class_labels: list of integer class IDs.
    """
    bboxes, class_labels = [], []

    if not os.path.exists(label_path):
        return bboxes, class_labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_labels.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:]])

    return bboxes, class_labels


def write_yolo_label(label_path: str, bboxes: list, class_labels: list) -> None:
    """
    Writes bounding boxes and class labels to a YOLO-format .txt file.

    Args:
        label_path:   Full path for the output .txt file.
        bboxes:       list of [x_center, y_center, width, height] (normalised).
        class_labels: list of integer class IDs.
    """
    with open(label_path, "w") as f:
        for cls, bbox in zip(class_labels, bboxes):
            x_c, y_c, w, h = bbox
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# Main augmentation function
# ---------------------------------------------------------------------------

def augment_training_data(
    images_dir: str,
    labels_dir: str,
    aug_images_dir: str,
    aug_labels_dir: str,
    augmentations_per_image: int = AUGMENTATIONS_PER_IMAGE,
) -> None:
    """
    Iterates over all training images, applies the augmentation pipeline
    N times per image, and saves the results with matching YOLO labels.

    Augmented files are named:  <original_stem>_aug<index>.jpg / .txt

    Args:
        images_dir:              Directory containing original training images.
        labels_dir:              Directory containing original YOLO .txt labels.
        aug_images_dir:          Directory to save augmented images.
        aug_labels_dir:          Directory to save augmented labels.
        augmentations_per_image: Number of augmented copies to generate per image.
    """
    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)

    pipeline = build_augmentation_pipeline()

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n{'='*60}")
    print(f"  Satellite Imagery Augmentation Pipeline")
    print(f"{'='*60}")
    print(f"  Source images      : {len(image_files)}")
    print(f"  Augmentations      : {augmentations_per_image} per image")
    print(f"  Total new samples  : {len(image_files) * augmentations_per_image}")
    print(f"{'='*60}\n")

    total_generated = 0
    total_skipped   = 0

    for img_file in image_files:
        img_stem = Path(img_file).stem
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, img_stem + ".txt")

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [WARN] Could not read image: {img_file}. Skipping.")
            total_skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read corresponding YOLO label
        bboxes, class_labels = read_yolo_label(lbl_path)
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
                    class_labels=class_labels,
                )

                aug_image  = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]

                # Skip if all bboxes were lost (object became invisible)
                if not aug_bboxes:
                    continue

                out_stem     = f"{img_stem}_aug{aug_idx}"
                out_img_path = os.path.join(aug_images_dir, out_stem + ".jpg")
                out_lbl_path = os.path.join(aug_labels_dir, out_stem + ".txt")

                # Save augmented image (convert back to BGR for OpenCV)
                cv2.imwrite(out_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

                # Save augmented YOLO label
                write_yolo_label(out_lbl_path, aug_bboxes, aug_labels)

                print(f"  Saved: {out_stem}.jpg  +  {out_stem}.txt")
                total_generated += 1

            except Exception as e:
                print(f"  [ERROR] Augmentation failed for {img_file} "
                      f"(aug {aug_idx}): {e}")

    print(f"\n  Augmentation complete.")
    print(f"  Images generated : {total_generated}")
    print(f"  Images skipped   : {total_skipped}")
    print(f"  New dataset size : original + {total_generated} samples\n")
