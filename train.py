import os
import sys
import json
import numpy as np
import skimage.io
from skimage.draw import polygon2mask
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Root directory of the project
ROOT_DIR = "C:\\Users\\azadk\\OneDrive\\Desktop\\projects\\masked_rcnn"

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs1")

# Configuration for training
class TruckConfig(Config):
    NAME = "truck"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # Background + Truck
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001

# Dataset loader for the truck dataset
class TruckDataset(utils.Dataset):
    def load_truck(self, dataset_dir, subset):
        # Add Truck class
        self.add_class("truck", 1, "Truck")

        # Train or validation dataset
        assert subset in ["train_car", "val_car"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations_path = os.path.join(dataset_dir, f"{subset}_json.json")
        with open(annotations_path) as f:
            annotations = json.load(f)

        # Add images
        for image_id, annotation in annotations.items():
            # Skip if no regions
            if not annotation['regions']:
                continue

            polygons = [r['shape_attributes'] for r in annotation['regions']]
            num_ids = [1] * len(polygons)  # All annotations are for the Truck class
            image_path = os.path.join(dataset_dir, annotation['filename'])

            try:
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
                continue

            self.add_image(
                "truck",
                image_id=annotation['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "truck":
            return super(self.__class__, self).load_mask(image_id)

        # Create a mask for each polygon
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            mask[..., i] = polygon2mask((info["height"], info["width"]),
                                        np.column_stack((p['all_points_y'], p['all_points_x'])))
        num_ids = np.array([1] * len(info["polygons"]), dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "truck":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)

# Training function
def train(model):
    dataset_train = TruckDataset()
    dataset_train.load_truck("C:\\Users\\azadk\\OneDrive\\Desktop\\projects\\masked_rcnn\\dataset", "train_car")
    dataset_train.prepare()

    dataset_val = TruckDataset()
    dataset_val.load_truck("C:\\Users\\azadk\\OneDrive\\Desktop\\projects\\masked_rcnn\\dataset", "val_car")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

# Instantiate configuration
config = TruckConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

# Load pre-trained weights
weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
    utils.download_trained_weights(COCO_WEIGHTS_PATH)

model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Start training
train(model)
