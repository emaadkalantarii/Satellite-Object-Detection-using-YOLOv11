# Import necessary libraries
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

# Define Paths
train_csv = r'dataset/labels/train.csv' # Training data CSV file
val_csv = r'dataset/labels/val.csv' # Validation data CSV file
train_images_dir = r'dataset/train' # Training images directory
val_images_dir = r'dataset/val' # Validation images directory
test_images_dir = r'dataset/test' # Test images directory
yaml_path = r'/mnt/aiongpfs/users/ekalantari/CVIA/dataset/data.yaml' # Path to YAML file




# Move the images from the different classes to the image folder.

# for test, train and val (function)
def move_images(directory):
    
    # Creates the directories to store the images and labels respectively as required per Yolo5
    try:
        os.mkdir(directory + "/images") 
    except FileExistsError:
        pass
    try:
        os.mkdir(directory + "/labels") 
    except FileExistsError:
        pass


    
    # Attempts to iterate through the directory to move the contained images to the /images folder.
    try:
        for filename in os.listdir(directory):

            # Ignores if the "file" is the folder "images" or "labels"
            if "images" in filename or "labels" in filename:
                print("Filename: ", filename)
                continue

            # Source
            f = os.path.join(directory, filename)
            print("Source:", f)

            # Destination
            dest = directory + "/images/" + filename
            print("Destination:", dest)

            # Moves the file
            os.rename(f, dest)
        
    # If a path does not exist,     
    except FileNotFoundError:
        print("File or folder not found. May have been processed already.")


move_images("dataset/train")
move_images("dataset/val")


class_number_pairs = {
    "smart_1": 0, 
    "cheops": 1, 
    "lisa_pathfinder": 2, 
    "debris": 3, 
    "proba_3_ocs": 4, 
    "soho": 5, 
    "earth_observation_sat_1": 6, 
    "proba_2": 7, 
    "xmm_newton": 8, 
    "double_star": 9, 
    "proba_3_csc": 10
    } 

# Open the train.csv labels.
with open(train_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    # For each entry / line in the CSV file.
    for row in csv_reader:

        # Skips the first row
        if 'bbox' in row:
            continue

        # Extracts the data "filename", class, and coords.
        img_name = row[0]
        img_class = row[1]
        img_coords = row[2]
        print(f'Filename:{img_name}\t Class:{img_class}\t Coordinates:{img_coords}.')

        # Creates a new .txt file at location dataset/train/labels/ with name = "img_name".
        img_name = "dataset/train/labels/" + img_name.split(".")[0] + ".txt"
        print("Text File name:", img_name)
        f = open(img_name, "w")
        img_coords = img_coords.replace('[', '')
        img_coords = img_coords.replace(']', '')
        img_coords = img_coords.split(", ")
        print("Coords:", img_coords)

        # Creates the string to be written in class x_center y_center width height format.
        img_label = img_class + " " + (" ").join(img_coords)
        print("Old Label:", img_label)

        row_min = int(img_coords[0])
        col_min = int(img_coords[1])
        row_max = int(img_coords[2])
        col_max = int(img_coords[3])

        print("Before:", row_min, col_min, row_max, col_max)

        # I summed the min and max values before doing the average to find the center. Then normalized by dividing by 1024. R = Y axis, C = X avis.
        x_center = round((((col_min + col_max) / 2) / 1024), 6)
        y_center = round((((row_min + row_max) / 2) / 1024), 6)

        # I did the difference and normalized by dividing by 1024, which is the fix width and height for all images in the dataset.
        width = abs(round(((col_max - col_min) / 1024), 6))
        height = abs(round(((row_max - row_min) / 1024), 6))

        print("After:", x_center, y_center, width, height)

        # Fixes image class that requires the int value.
        print("Image Class before:", img_class)
        img_class = class_number_pairs[img_class]
        print("Image Class after:", img_class)

        img_label = str(img_class) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
        print("New Label:", img_label)
        # Write the new label as a one line string and closes the file.
        f.write(img_label)
        f.close()

# Open the val.csv labels.
with open(val_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    # For each entry / line in the CSV file.
    for row in csv_reader:

        # Skips the first row
        if 'bbox' in row:
            continue

        # Extracts the data "filename", class, and coords.
        img_name = row[0]
        img_class = row[1]
        img_coords = row[2]
        print(f'Filename:{img_name}\t Class:{img_class}\t Coordinates:{img_coords}.')

        # Creates a new .txt file at location dataset/val/labels/ with name = "img_name".
        img_name = "dataset/val/labels/" + img_name.split(".")[0] + ".txt"
        print("Text File name:", img_name)
        f = open(img_name, "w")
        img_coords = img_coords.replace('[', '')
        img_coords = img_coords.replace(']', '')
        img_coords = img_coords.split(", ")
        print("Coords:", img_coords)

        # Creates the string to be written in class x_center y_center width height format.
        img_label = img_class + " " + (" ").join(img_coords)
        print("Old Label:", img_label)

        row_min = int(img_coords[0])
        col_min = int(img_coords[1])
        row_max = int(img_coords[2])
        col_max = int(img_coords[3])

        print("Before:", row_min, col_min, row_max, col_max)

        # I summed the min and max values before doing the average to find the center. Then normalized by dividing by 1024. R = Y axis, C = X avis.
        x_center = round((((col_min + col_max) / 2) / 1024), 6)
        y_center = round((((row_min + row_max) / 2) / 1024), 6)

        # I did the difference and normalized by dividing by 1024, which is the fix width and height for all images in the dataset.
        width = abs(round(((col_max - col_min) / 1024), 6))
        height = abs(round(((row_max - row_min) / 1024), 6))
        
        print("After:", x_center, y_center, width, height)

        # Fixes image class that requires the int value.
        print("Image Class before:", img_class)
        img_class = class_number_pairs[img_class]
        print("Image Class after:", img_class)

        img_label = str(img_class) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height)
        print("New Label:", img_label)
        # Write the new label as a one line string and closes the file.
        f.write(img_label)
        f.close()
        



# Train the YOLOv11 Model
model = YOLO("yolo11m.pt")
model.train(
    data=CVIA/dataset/data.yaml,
    epochs=35,             # Increase epochs for better training
    imgsz=1024,            # Larger image size for better accuracy
    batch=8,              # Larger batch size (hardware-dependent)
    optimizer='auto',      # Optimizer choice
    lr0=0.01,             # Initial learning rate
    weight_decay=0.0005,   # L2 regularization to prevent overfitting
    patience=100,           # Early stopping patience
    augment=True,          # Use augmentation
    device=0,              # GPU (use 'cpu' if GPU unavailable)
    workers=4,             # Number of data-loading workers
    save_period=1,         # Save model after every epoch
    project='runs/model_m',  # Directory to save training results
    name='model_m_35_4',    # Experiment name
    verbose=True           # Verbose output
)



