# Truck Detection using Mask R-CNN

This project leverages Mask R-CNN, a powerful object detection and instance segmentation model, to detect trucks in images. It includes both training and testing phases to create a custom-trained model capable of recognizing trucks with high accuracy.

---

## Features
- **Custom Dataset Training**: Train a model using a dataset annotated with truck regions.
- **Instance Segmentation**: Outputs include the bounding boxes, segmentation masks, and class probabilities for trucks.
- **Interactive Testing**: Use an interactive GUI for testing the trained model on new images.
- **Visualization**: Results are displayed with bounding boxes and masks overlaid on the original images.

---

## Repository Structure
```
masked_rcnn/
├── dataset/                # Dataset directory
│   ├── train_car/          # Training data
│   ├── val_car/            # Validation data
├── logs1/                  # Logs and trained model weights
├── mask_rcnn_coco.h5       # Pre-trained COCO weights
├── train.py                # Training script
├── test.py                 # Testing script with GUI
└── README.md               # Documentation
```

---

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- SciPy
- Scikit-Image
- Matplotlib
- Tkinter (for GUI testing)
- Mask R-CNN Library (from Matterport)

To install the required Python packages, run:
```bash
pip install -r requirements.txt
```

---

## Dataset
The dataset must be structured into `train_car` and `val_car` directories, containing:
- Image files (e.g., `.jpg`, `.png`)
- Corresponding JSON annotation files with polygon coordinates for segmentation masks.

### Annotation Format
Annotations should follow this structure:
```json
{
    "image_id": {
        "filename": "image_name.jpg",
        "regions": [
            {
                "shape_attributes": {
                    "all_points_x": [x1, x2, ...],
                    "all_points_y": [y1, y2, ...]
                }
            }
        ]
    }
}
```

---

## Training

### Steps:
1. **Prepare the Dataset**: Ensure the dataset is organized as per the specified format.
2. **Configure Training**: Edit the `TruckConfig` class in the `train.py` file to match your system and dataset parameters.
3. **Run Training**:
   ```bash
   python train.py
   ```
   
### Training Script Overview (`train.py`):
- Loads the dataset and prepares it for training.
- Configures the `Mask R-CNN` model.
- Trains the model using the COCO pre-trained weights as the starting point.
- Saves the trained weights in the `logs1/` directory.

---

## Testing

### Steps:
1. **Select an Image**: Use the GUI to select an image for testing.
2. **Run Testing**:
   ```bash
   python test.py
   ```

### Testing Script Overview (`test.py`):
- Loads the trained model weights.
- Uses the Mask R-CNN model in inference mode.
- Allows the user to select an image via a file dialog.
- Runs object detection and displays the results with bounding boxes and segmentation masks.

---

## Output Results
### Example Results

#### Input Images
1. **Front View**
   ![Front View Input](images/front_view_input.jpg)
2. **Side View**
   ![Side View Input](images/side_view_input.jpg)

#### Output Results
1. **Front View Detection**
   ![Front View Output](images/front_view_output.jpg)
2. **Side View Detection**
   ![Side View Output](images/side_view_output.jpg)

---

## Configuration
### Training Configuration
Edit the `TruckConfig` class in `train.py` for custom parameters:
```python
class TruckConfig(Config):
    NAME = "truck"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # Background + Truck
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001
```

### Testing Configuration
The `TruckConfig` class in `test.py` mirrors the training configuration:
```python
class TruckConfig(Config):
    NAME = "truck"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + Truck
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9
```

---

## Pre-trained Weights
The training process starts with the COCO pre-trained weights (`mask_rcnn_coco.h5`). These weights can be downloaded from the [official Mask R-CNN GitHub repository](https://github.com/matterport/Mask_RCNN/releases).

---

## Additional Notes
- Ensure that all dependencies are installed and compatible.
- The `logs1/` directory will store the trained model weights, which are loaded during testing.
- Modify paths in the scripts as per your project structure.

---

## Acknowledgments
- [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN)
- [COCO Dataset](https://cocodataset.org/)

---


