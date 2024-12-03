import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib

# Root directory of the project
ROOT_DIR = "C:/Users/azadk/OneDrive/Desktop/projects/masked_rcnn"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs1")
MODEL_DIR = os.path.join(ROOT_DIR, "logs1")

# Path to the trained weights for the truck model
WEIGHTS_PATH = "C:/Users/azadk/OneDrive/Desktop/projects/masked_rcnn/logs1/truck20241201T2301/mask_rcnn_truck_0010.h5"  # Update if necessary

class TruckConfig(Config):
    """Configuration for detecting trucks."""
    NAME = "truck"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + Truck
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9

# Dataset class for truck identification
class TruckDataset(utils.Dataset):
    def load_truck(self, dataset_dir, subset):
        """Load the truck dataset."""
        self.add_class("truck", 1, "Truck")
        assert subset in ["train_car", "val_car"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations_path = os.path.join(dataset_dir, f"{subset}_json.json")
        annotations = json.load(open(annotations_path))
        annotations = [a for a in annotations.values() if a['regions']]

        # Add images
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            image_path = os.path.join(dataset_dir, a['filename'])
            try:
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
                continue

            self.add_image(
                "truck",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "truck":
            return super(self.__class__, self).load_mask(image_id)

        # Create mask
        masks = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            masks[rr, cc, i] = 1
        return masks, np.ones([masks.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "truck":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)

# Function to set up visualization axes
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# Load validation dataset
CUSTOM_DIR = "C:/Users/azadk/OneDrive/Desktop/projects/masked_rcnn/dataset"
dataset = TruckDataset()
dataset.load_truck(CUSTOM_DIR, "val_car")
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Configuration
config = TruckConfig()

# Load Mask R-CNN model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights for the trained truck model
print("Loading weights from ", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name=True)

# Function to open file dialog for image selection
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    return file_path

# Use the file dialog to allow the user to select an image
path_to_new_image = select_image()

if path_to_new_image:  # Check if the user selected an image
    print(f"Selected image: {path_to_new_image}")
    
    # Read and process the image
    image = mpimg.imread(path_to_new_image)

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                ["BG", "Truck"], r['scores'], title="Truck Detection")
else:
    print("No image selected.")
