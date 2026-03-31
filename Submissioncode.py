"""
Submissioncode.py
------------------
Inference script for satellite object detection using a fine-tuned
YOLOv11-Medium model.

Runs predictions on all test images and writes results to a CSV file
with columns: filename, class, bbox (x1,y1,x2,y2).
"""

import os
import csv
import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths  —  update these to match your local environment
# ---------------------------------------------------------------------------
MODEL_PATH      = "runs/model_m/model_m_35/weights/best.pt"
TEST_IMAGES_DIR = "dataset/test"
OUTPUT_CSV_PATH = "dataset/submission.csv"

# Minimum confidence score to accept a detection
CONFIDENCE_THRESHOLD = 0.25

# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Class mapping  (id → name)
# ---------------------------------------------------------------------------
CLASS_ID_TO_NAME = {
    0:  "smart_1",
    1:  "cheops",
    2:  "lisa_pathfinder",
    3:  "debris",
    4:  "proba_3_ocs",
    5:  "soho",
    6:  "earth_observation_sat_1",
    7:  "proba_2",
    8:  "xmm_newton",
    9:  "double_star",
    10: "proba_3_csc",
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict(image_path: str, model: YOLO) -> tuple[str, str, list[int]]:
    """
    Runs inference on a single image and returns the highest-confidence
    detection.

    Args:
        image_path: Absolute or relative path to the image file.
        model:      Loaded YOLO model instance.

    Returns:
        Tuple of (filename, class_name, [x1, y1, x2, y2]).
        Returns ("unknown", [0, 0, 0, 0]) if no detection is found.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image, conf=CONFIDENCE_THRESHOLD)
    detections = results[0].boxes

    filename = os.path.basename(image_path)

    if len(detections) == 0:
        return filename, "unknown", [0, 0, 0, 0]

    # Sort detections by confidence (descending) and take the best one
    confidences = detections.conf.tolist()
    best_idx = confidences.index(max(confidences))
    best = detections[best_idx]

    x1, y1, x2, y2 = best.xyxy[0].tolist()
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    class_name = CLASS_ID_TO_NAME.get(int(best.cls[0]), "unknown")

    return filename, class_name, bbox


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.eval()

    image_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    print(f"Found {len(image_files)} test images in '{TEST_IMAGES_DIR}'")

    with open(OUTPUT_CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "class", "bbox"])

        for image_name in image_files:
            image_path = os.path.join(TEST_IMAGES_DIR, image_name)
            try:
                filename, class_name, bbox = predict(image_path, model)
                bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                writer.writerow([filename, class_name, bbox_str])
                print(f"  ✓ {filename}  →  {class_name}  {bbox_str}")
            except Exception as e:
                print(f"  ✗ Error processing {image_name}: {e}")
                writer.writerow([image_name, "unknown", "0,0,0,0"])

    print(f"\nSubmission saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
