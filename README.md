# Object-detection-MRCNN

## Project Overview
This project uses the **Mask R-CNN** framework for detecting and segmenting trucks in images. The project includes:
1. **Truck-specific Configuration**: Custom settings optimized for truck detection.
2. **Dataset Loader**: Load and preprocess annotated images for training and validation.
3. **Mask Generation**: Create segmentation masks for truck regions in the images.
4. **Training Pipeline**: Train the model using a pre-trained Mask R-CNN model with COCO weights as a starting point.

---

## Installation

### Prerequisites
Ensure the following libraries and tools are installed:
- Python 3.7+
- TensorFlow (compatible with Mask R-CNN)
- Keras
- NumPy
- scikit-image
- OpenCV (optional, for image processing)
- h5py
- Matplotlib

Install the required Python packages:
```bash
pip install numpy tensorflow keras scikit-image opencv-python matplotlib h5py
```

### Clone the Repository
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/mask-rcnn-truck-detection.git
cd mask-rcnn-truck-detection
```

### Download Pre-Trained Weights
Download the pre-trained **COCO weights** for Mask R-CNN:
```bash
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_coco.h5
```
Place the weights file in the project root directory.

---

## Dataset Structure
The dataset should be organized as follows:
```
dataset/
│
├── train_car/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│   └── train_car_json.json
│
└── val_car/
    ├── image1.jpg
    ├── image2.jpg
    ├── ...
    └── val_car_json.json
```

### Annotation Format
- Each annotation JSON file contains regions with `shape_attributes` (polygon points) and metadata for images.
- Example annotation format:
```json
{
  "image1.jpg": {
    "filename": "image1.jpg",
    "regions": [
      {
        "shape_attributes": {
          "all_points_x": [x1, x2, x3, ...],
          "all_points_y": [y1, y2, y3, ...]
        }
      },
      ...
    ]
  }
}
```

---

## Usage

### Configuration
The project uses the `TruckConfig` class, which specifies:
- **Name**: `truck`
- **Images per GPU**: `2`
- **Number of Classes**: `2` (Background + Truck)
- **Steps per Epoch**: `10`
- **Detection Confidence**: `0.9`
- **Learning Rate**: `0.001`

### Running the Training
Run the script to start training:
```bash
python Custom1.py
```
This script:
1. Loads and preprocesses the dataset.
2. Initializes the Mask R-CNN model with the pre-trained COCO weights.
3. Trains the network heads on the truck dataset.

---

## Output
### Model Training Logs
Training logs, including checkpoints and metrics, are saved in the `logs1/` directory.

### Detection and Segmentation Results
The trained model can segment truck regions from input images, creating masks around detected trucks. The masks represent the segmented truck regions.

---

## Future Enhancements
- **Augment Dataset**: Add more annotated images for better generalization.
- **Fine-tune Hyperparameters**: Experiment with learning rates, batch sizes, and epochs.
- **Deployment**: Create a script for real-time truck detection using the trained model.

---

## Acknowledgments
- **Matterport's Mask R-CNN Framework**: [GitHub Repository](https://github.com/matterport/Mask_RCNN)
- **COCO Dataset**: Pre-trained weights used as the base model.

Feel free to contribute or raise issues for this project! 🚀
