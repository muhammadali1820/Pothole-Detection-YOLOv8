# Pothole Detection using YOLOv8

This project implements a deep learning model for detecting potholes in road images using the YOLOv8 object detection algorithm. The model is trained on a custom dataset of road images containing potholes and achieves high accuracy in detecting potholes in real-world scenarios.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training Process](#training-process)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Experiments](#experiments)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

## Overview

Road infrastructure monitoring is crucial for maintaining safe driving conditions. Potholes pose significant risks to vehicles and drivers, making automated detection systems valuable for municipal authorities and transportation departments. This project leverages state-of-the-art computer vision techniques to automatically detect and localize potholes in road images.

### Key Features
- Real-time pothole detection
- High accuracy (mAP50 up to 78.3%)
- Multiple model variants tested
- Comprehensive evaluation metrics
- Easy deployment-ready format

## Dataset

The model was trained on a custom pothole detection dataset obtained from Roboflow. The dataset contains:
- **Training images**: 1,939
- **Validation images**: 555  
- **Test images**: 277
- **Total images**: 2,771
- **Classes**: 1 (Pothole)
- **Image formats**: JPG/PNG

The dataset was split into train/validation/test sets with a ratio designed to ensure robust model evaluation.

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Pothole-Detection-YOLOv8.git
cd Pothole-Detection-YOLOv8
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install ultralytics opencv-python matplotlib torch torchvision
```

4. Download the trained model weights from releases or train your own model.

## Training Process

The training process involves several steps:

1. **Data Preparation**: The dataset is downloaded and prepared using Roboflow integration
2. **Model Selection**: YOLOv8n (nano) and YOLOv8s (small) models were used
3. **Training Configuration**: Custom hyperparameters were set for optimal performance
4. **Model Training**: Training was performed with different configurations

### Training Parameters
- **Epochs**: 20 (initial), 50 (optimized)
- **Image Size**: 640 (initial), 832 (optimized) 
- **Batch Size**: 16 (initial), 12 (for larger images)
- **Optimizer**: Auto-selected (AdamW)
- **Device**: GPU (if available)

## Model Architecture

The project utilizes the YOLOv8 architecture, which is known for its efficiency and accuracy in real-time object detection. Two variants were tested:

1. **YOLOv8n (Nano)**: Lightweight model suitable for edge devices
2. **YOLOv8s (Small)**: Slightly larger model with improved accuracy

Both models use the same underlying architecture with different scaling factors for depth and width.

## Results

### Baseline Model (YOLOv8n, 20 epochs, 640 image size)
- **mAP50**: 73.9%
- **Precision**: 76.4%
- **Recall**: 64.8%
- **mAP50-95**: 42.3%

### Optimized Model (YOLOv8s, 50 epochs, 832 image size)
- **mAP50**: 78.3%
- **Precision**: 76.3%
- **Recall**: 71.5%
- **mAP50-95**: 45.8%

The optimized model showed a 4.4% improvement in mAP50 compared to the baseline.

## Experiments

### Experiment 1: Extended Training
- Increased epochs from 20 to 50
- Increased image size from 640 to 832
- Result: Improved mAP50 to 76.9%

### Experiment 2: Larger Model
- Used YOLOv8s instead of YOLOv8n
- 50 epochs with 832 image size
- Result: Achieved 78.3% mAP50

## Usage

### Using Pre-trained Models

```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('best.pt')

# Perform inference on an image
results = model('path/to/image.jpg')

# Display results
results.show()

# Or save results
results.save()
```

### Custom Inference

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('best.pt')

# Run inference on a video
results = model('path/to/video.mp4', save=True)

# Process results list
for r in results:
    # Process each result
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segmentation masks outputs
    probs = r.probs  # Probs object for classification outputs
```

## Performance Metrics

The model was evaluated using standard object detection metrics:

- **mAP50**: Mean Average Precision at IoU threshold of 0.50
- **mAP50-95**: Mean Average Precision at IoU thresholds from 0.50 to 0.95
- **Precision**: Ratio of true positives to total predicted positives
- **Recall**: Ratio of true positives to total actual positives
- **F1-Score**: Harmonic mean of precision and recall

### Model Comparison

| Model Configuration | mAP50 | Precision | Recall | mAP50-95 |
|---------------------|-------|-----------|--------|----------|
| YOLOv8n (Baseline) | 73.9% | 76.4% | 64.8% | 42.3% |
| YOLOv8n (Exp1) | 76.9% | 77.1% | 69.4% | 45.5% |
| YOLOv8s (Exp2) | 78.3% | 76.3% | 71.5% | 45.8% |

## Model Files

The repository includes:
- `best.pt`: The best performing model weights (YOLOv8s variant)
- `best (1).pt`: Alternative model weights
- `best (2).pt`: Additional model weights
- `Untitled0.ipynb`: Jupyter notebook with complete training code

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- Roboflow for the dataset preparation tools
- The open-source computer vision community

## Contact

For questions or support, please open an issue in the repository.