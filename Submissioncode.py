# Load the YOLOv11 model
# Import Libraries
import os
import pandas as pd
import numpy as np
import shutil
import cv2
import random
import albumentations as A
from ultralytics import YOLO
import csv
from PIL import Image
import torch


model = YOLO("/mnt/aiongpfs/users/ekalantari/CVIA/runs/model_m_35/best.pt")
model.eval()  # Set the model to evaluation mode

# Class mapping dictionary
class_pairs = {
    0: "smart_1",
    1: "cheops",
    2: "lisa_pathfinder",
    3: "debris",
    4: "proba_3_ocs",
    5: "soho",
    6: "earth_observation_sat_1",
    7: "proba_2",
    8: "xmm_newton",
    9: "double_star",
    10: "proba_3_csc"
}

# Map the predicted label index to the class name
def class_mapping(class_number):
    return class_pairs.get(class_number, "unknown")

# Perform inference and get predictions
def predict_image_class_and_bbox(image_path, model, confidence_threshold=0.25):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Perform inference
    results = model(image)

    # Extract bounding boxes and class labels
    detections = results[0].boxes  # Access detected boxes
    if len(detections) == 0:
        return os.path.basename(image_path), "unknown", [0, 0, 0, 0]

    # Use the detection with the highest confidence
    best_detection = detections[0]
    x1, y1, x2, y2 = best_detection.xyxy[0].tolist()
    cls = int(best_detection.cls[0])

    # Convert bounding box coordinates to integers
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    class_name = class_mapping(cls)

    return os.path.basename(image_path), class_name, bbox

# Path to the test images directory
test_images_dir = "/mnt/aiongpfs/users/ekalantari/CVIA/dataset/test"

# Output CSV file path
output_csv_path = "/mnt/aiongpfs/users/ekalantari/CVIA/dataset/submission.csv"

# Create the submission CSV file
with open(output_csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "class", "bbox"])  # Write the header

    # Iterate through all images in the test directory
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)

        # Check if the file is an image
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_name}")
            continue

        try:
            # Predict the class and bounding box for the image
            filename, predicted_class, bbox = predict_image_class_and_bbox(image_path, model)
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            writer.writerow([filename, predicted_class, bbox_str])
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            writer.writerow([image_name, "unknown", "0,0,0,0"])

print(f"Submission file created at: {output_csv_path}")