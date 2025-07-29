# Unity-Perception-to-YOLO

Convert Unity Perception synthetic datasets to YOLO format and train YOLOv8 object detection models.

## Overview

This project demonstrates how to train modern computer vision models using purely synthetic data generated with Unity's Perception package. The pipeline converts Unity's SOLO format to YOLO format and trains YOLOv8 models for real-world object detection.

The model has been trained on this synthetic dataset:
https://github.com/BjoernGit/PerceptionAutoLabeling

## Training Demo

![Training Process](./Videos/KobraTrained.gif)

## Features

- **Flexible SOLO to YOLO Converter**: Supports both Unity Perception output formats
- **YOLOv8 Training Pipeline**: Complete training script with optimized hyperparameters
- **Real-time Webcam Testing**: Live detection with adjustable confidence thresholds
- **Synthetic to Real-World**: Proven pipeline for domain transfer

## Quick Start

### 1. Convert Unity Dataset
```bash
python src/flexible_solo_converter.py "path/to/unity/solo_data" "yolo_dataset"
```

### 2. Train YOLOv8 Model
```bash
# Small model (fast training)
python src/train.py --model n --epochs 50 --batch 16 --dataset yolo_dataset

# Large model (better accuracy)
python src/train.py --model x --epochs 100 --batch 8 --dataset yolo_dataset
```

### 3. Test with Webcam
```bash
python src/webcam_detect.py --model runs/train/unity_perception/weights/best.pt --conf 0.8
```

## Scripts

- **`flexible_solo_converter.py`**: Converts Unity Perception SOLO format to YOLO
- **`train.py`**: Complete YOLOv8 training pipeline with synthetic data optimizations
- **`webcam_detect.py`**: Real-time webcam detection with confidence adjustment

## Model Sizes

| Model | Parameters | Speed | Accuracy |
|-------|------------|-------|----------|
| YOLOv8n | 3.2M | Fastest | Good |
| YOLOv8s | 11.2M | Fast | Better |
| YOLOv8m | 25.9M | Medium | Great |
| YOLOv8l | 43.7M | Slow | Excellent |
| YOLOv8x | 68.2M | Slowest | Best |

## Results

Successfully trained models achieve strong real-world performance despite being trained exclusively on synthetic Unity data, demonstrating effective sim-to-real transfer learning.
