# Unity-Perception-to-YOLO

Convert Unity Perception synthetic datasets to YOLO format and train YOLO11 object detection models with an intuitive GUI.

## Overview

This project demonstrates how to train modern computer vision models using purely synthetic data generated with Unity's Perception package. The pipeline converts Unity's SOLO format to YOLO format and provides both command-line and GUI interfaces for training YOLO11 models for real-world object detection.
The model has been trained on this synthetic dataset:
https://github.com/BjoernGit/PerceptionAutoLabeling

## Training Demo

![Training Process](./Videos/KobraTrained.gif)

## Features

🎯 YOLO11 Training GUI: Interactive interface with real-time configuration and augmentation preview
🔄 Flexible SOLO to YOLO Converter: Supports both Unity Perception output formats
📊 Augmentation Visualizer: Live preview of data augmentations before training
📹 Real-time Webcam Testing: Live detection with adjustable confidence thresholds
🔬 Synthetic to Real-World: Pipeline for domain transfer
💾 Configuration Management: Save and load training configurations

## Quick Start

### 1. Convert Unity Dataset
```bash
python src/flexible_solo_converter.py "path/to/unity/solo_data" "yolo_dataset"
```

### 2. Train YOLOv11 Model
```bash
# Launch the interactive training GUI
python training_gui.py

#  Or train YOLO11 model in command line
python src/train.py --model m --epochs 100 --batch 16 --dataset yolo_dataset

```

### 3. Test with Webcam
```bash
python src/webcam_detect.py --model runs/train/unity_perception/weights/best.pt --conf 0.8
```

## Scripts

- **`flexible_solo_converter.py`**: Converts Unity Perception SOLO format to YOLO
- **`train.py`**: Complete YOLOv11 training pipeline with synthetic data optimizations
- - **`training_gui.py`**: Interactive GUI for YOLO11 training with augmentation preview
- **`webcam_detect.py`**: Real-time webcam detection with confidence adjustment

## Model Sizes

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| YOLO11n | 2.6M | Fastest | Good | Mobile/Edge |
| YOLO11s | 9.4M | Fast | Better | Real-time |
| YOLO11m | 20.1M | Medium | Great | Balanced |
| YOLO11l | 25.3M | Slow | Excellent | High Accuracy |
| YOLO11x | 56.9M | Slowest | Best | Maximum Performance |

## Interface

### Main Settings
Configure dataset paths, model sizes, and core training parameters with an intuitive layout.
<img width="797" height="523" alt="MainSettings" src="https://github.com/user-attachments/assets/64e0bce2-46eb-4455-8599-ac5355d81358" />

### Augmentation Controls
Fine-tune geometric transforms, blur effects, and augmentation probabilities with real-time sliders.
<img width="927" height="1071" alt="Augmentation" src="https://github.com/user-attachments/assets/f3cee8d7-cea2-47a8-9f2f-7e4bc4886551" />

### Live Augmentation Preview
Visualize augmentation effects on sample images before training with customizable grid layouts.
<img width="1185" height="1269" alt="Preview" src="https://github.com/user-attachments/assets/83c46f64-7e7b-4285-a6a3-0804b26703d6" />

### Training Monitor
Monitor real-time training progress with command preview and live output streaming.
<img width="1132" height="724" alt="Training" src="https://github.com/user-attachments/assets/bf52b5ce-da5f-40b2-bbb6-bb55027b1d71" />

## Results

Successfully trained models achieve strong real-world performance despite being trained exclusively on synthetic Unity data, demonstrating effective sim-to-real transfer learning.
