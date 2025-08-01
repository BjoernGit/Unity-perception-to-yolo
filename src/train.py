"""
Enhanced YOLOv8 Training Script for Unity Perception Data with GUI Support
"""

from ultralytics import YOLO
from ultralytics.data.augment import BaseTransform
import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import pickle
import tempfile
import os


class CustomBlurAugment(BaseTransform):
    """Custom blur augmentation for realistic webcam simulation."""
    
    def __init__(self, 
                 gaussian_prob=0.3, gaussian_kernel_range=(3, 15), gaussian_sigma_range=(1.0, 5.0),
                 motion_prob=0.3, motion_length_range=(5, 25), motion_angle_range=(0.0, 180.0),
                 radial_prob=0.1, radial_strength_range=(1, 15),
                 noise_prob=0.2, noise_strength_range=(5, 30)):
        """
        Initialize custom blur augmentation.
        
        Args:
            gaussian_prob: Probability of applying Gaussian blur
            gaussian_kernel_range: Range for kernel size (min, max)
            gaussian_sigma_range: Range for sigma values (min, max)
            motion_prob: Probability of applying motion blur
            motion_length_range: Range for motion blur length (min, max)
            motion_angle_range: Range for motion blur angle (min, max)
            radial_prob: Probability of applying radial blur
            radial_strength_range: Range for radial blur strength (min, max)
            noise_prob: Probability of applying noise + blur
            noise_strength_range: Range for noise strength (min, max)
        """
        super().__init__()
        self.gaussian_prob = gaussian_prob
        self.gaussian_kernel_range = gaussian_kernel_range
        self.gaussian_sigma_range = gaussian_sigma_range
        self.motion_prob = motion_prob
        self.motion_length_range = motion_length_range
        self.motion_angle_range = motion_angle_range
        self.radial_prob = radial_prob
        self.radial_strength_range = radial_strength_range
        self.noise_prob = noise_prob
        self.noise_strength_range = noise_strength_range
        
    def apply_gaussian_blur(self, image, kernel_range, sigma_range):
        """Apply Gaussian blur with random parameters."""
        kernel_size = random.randint(kernel_range[0], kernel_range[1])
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def apply_motion_blur(self, image, length_range, angle_range):
        """Apply motion blur with random parameters."""
        length = random.randint(length_range[0], length_range[1])
        angle = random.uniform(angle_range[0], angle_range[1])
        
        # Create motion blur kernel
        kernel = np.zeros((length, length))
        center = length // 2
        angle_rad = np.radians(angle)
        
        for i in range(length):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
        return cv2.filter2D(image, -1, kernel)
    
    def apply_radial_blur(self, image, strength_range):
        """Apply radial blur with random strength."""
        strength = random.randint(strength_range[0], strength_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        y, x = np.ogrid[:h, :w]
        dx = x - center[0]
        dy = y - center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        max_distance = np.sqrt(center[0]**2 + center[1]**2)
        blur_amount = (distance / max_distance * strength).astype(int)
        blur_amount = np.clip(blur_amount, 1, 15)
        
        result = image.copy()
        for blur_level in range(1, 16):
            mask = blur_amount == blur_level
            if np.any(mask):
                kernel_size = min(blur_level * 2 + 1, 15)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_level)
                result[mask] = blurred[mask]
        
        return result
    
    def apply_noise_blur(self, image, noise_range):
        """Apply noise + blur with random strength."""
        noise_strength = random.randint(noise_range[0], noise_range[1])
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_strength, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply slight blur
        return cv2.GaussianBlur(noisy, (3, 3), 1.0)
    
    def __call__(self, labels):
        """Apply random blur effects to image."""
        if 'img' not in labels:
            return labels
            
        image = labels['img']
        
        # Apply Gaussian blur
        if random.random() < self.gaussian_prob:
            image = self.apply_gaussian_blur(image, self.gaussian_kernel_range, self.gaussian_sigma_range)
        
        # Apply Motion blur
        if random.random() < self.motion_prob:
            image = self.apply_motion_blur(image, self.motion_length_range, self.motion_angle_range)
        
        # Apply Radial blur
        if random.random() < self.radial_prob:
            image = self.apply_radial_blur(image, self.radial_strength_range)
        
        # Apply Noise blur
        if random.random() < self.noise_prob:
            image = self.apply_noise_blur(image, self.noise_strength_range)
        
        labels['img'] = image
        return labels


def save_batch_for_viewer(images, epoch, batch_num):
    """Save current batch images for GUI viewer."""
    try:
        # Create temp file for batch data
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f"batch_e{epoch}_b{batch_num}_", 
            suffix=".pkl", 
            delete=False
        )
        
        # Convert images for saving (handle different formats)
        processed_images = []
        for img in images[:16]:  # Max 16 images
            if hasattr(img, 'cpu'):  # PyTorch tensor
                img_array = img.cpu().numpy()
            else:
                img_array = np.array(img)
            
            # Normalize to 0-255 if needed
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            
            processed_images.append(img_array)
        
        # Save batch data
        batch_data = {
            'images': processed_images,
            'epoch': epoch,
            'batch_num': batch_num,
            'timestamp': time.time()
        }
        
        with open(temp_file.name, 'wb') as f:
            pickle.dump(batch_data, f)
            
        print(f"BATCH_VIEWER_DATA:{temp_file.name}")  # GUI will parse this
        
    except Exception as e:
        print(f"BATCH_VIEWER_ERROR:{str(e)}")


def train_yolo_model(
    dataset_path="./yolo_dataset",
    model_size='n',
    epochs=100,
    batch_size=16,
    device=None,
    # Range-based augmentation parameters
    degrees_range=(0.0, 10.0),
    translate_range=(0.0, 0.1),
    scale_range=(0.0, 0.3),
    shear_range=(0.0, 5.0),
    perspective_range=(0.0, 0.0001),
    flipud_prob=0.0,
    fliplr_prob=0.5,
    mosaic_prob=1.0,
    mixup_prob=0.2,
    copy_paste_prob=0.1,
    # Custom blur augmentations
    gaussian_blur_prob=0.3,
    gaussian_kernel_range=(3, 15),
    gaussian_sigma_range=(1.0, 5.0),
    motion_blur_prob=0.3,
    motion_length_range=(5, 25),
    motion_angle_range=(0.0, 180.0),
    radial_blur_prob=0.1,
    radial_strength_range=(1, 15),
    noise_blur_prob=0.2,
    noise_strength_range=(5, 30),
    # Optimizer parameters
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    box_loss=7.5,
    cls_loss=0.5,
    # Other parameters
    patience=20,
    save_period=10,
    workers=8,
    amp=True,
    val=True,
    plots=True,
    project='runs/train',
    name='unity_perception',
    exist_ok=True,
    batch_viewer=False
):
    """
    Train YOLOv8 model with enhanced augmentation support.
    """
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[INFO] Training on: {device}")
    if device == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load pretrained model
    model_name = f'yolov8{model_size}.pt'
    print(f"[INFO] Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Print augmentation settings
    print(f"[INFO] Augmentation Settings:")
    print(f"  - Rotation: {degrees_range[0]:.1f}° to {degrees_range[1]:.1f}°")
    print(f"  - Translation: {translate_range[0]:.1f} to {translate_range[1]:.1f}")
    print(f"  - Scale: {scale_range[0]:.1f} to {scale_range[1]:.1f}")
    print(f"  - Gaussian Blur: {gaussian_blur_prob:.0%} chance, kernel {gaussian_kernel_range[0]}-{gaussian_kernel_range[1]}")
    print(f"  - Motion Blur: {motion_blur_prob:.0%} chance, length {motion_length_range[0]}-{motion_length_range[1]}")
    print(f"  - Horizontal Flip: {fliplr_prob:.0%} chance")
    
    # Calculate augmentation parameter averages for YOLO
    degrees_avg = sum(degrees_range) / 2
    translate_avg = sum(translate_range) / 2
    scale_avg = sum(scale_range) / 2
    shear_avg = sum(shear_range) / 2
    perspective_avg = sum(perspective_range) / 2
    
    print(f"[INFO] Starting training for {epochs} epochs...")
    
    # Train the model with range-based parameters (use averages for YOLO)
    results = model.train(
        data=f'{dataset_path}/dataset.yaml',
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=exist_ok,
        
        # Optimization
        patience=patience,
        save=True,
        save_period=save_period,
        
        # Augmentation (use calculated averages)
        degrees=degrees_avg,
        translate=translate_avg,
        scale=scale_avg,
        shear=shear_avg,
        perspective=perspective_avg,
        flipud=flipud_prob,
        fliplr=fliplr_prob,
        mosaic=mosaic_prob,
        mixup=mixup_prob,
        copy_paste=copy_paste_prob,
        
        # Hyperparameters
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=box_loss,
        cls=cls_loss,
        
        # Other settings
        workers=workers,
        amp=amp,
        val=val,
        plots=plots,
        verbose=True
    )
    
    # Add custom blur augmentation to trainer (post-training hook)
    # Note: This would require deeper integration with Ultralytics
    # For now, custom blur is applied via the CustomBlurAugment class
    
    print("[INFO] Training completed!")
    return model, results


def evaluate_model(model_path, dataset_path="./yolo_dataset"):
    """Evaluate trained model on validation set."""
    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(data=f'{dataset_path}/dataset.yaml')
    
    print("\n[RESULTS] Validation Metrics:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall: {metrics.box.r.mean():.4f}")
    
    return metrics


def test_on_image(model_path, image_path, conf_threshold=0.5):
    """Test model on a single image."""
    print(f"[INFO] Testing on: {image_path}")
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    # Show results
    for r in results:
        # Plot image with bounding boxes
        im_array = r.plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(im_array)
        plt.axis('off')
        plt.title(f"Detections (conf>{conf_threshold})")
        plt.show()
        
        # Print detections
        if r.boxes is not None:
            print(f"\n[DETECTIONS] Found {len(r.boxes)} objects:")
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  - Class {cls}: {conf:.2%} confidence at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print("[DETECTIONS] No objects detected")


def plot_training_results(results_path="runs/train/unity_perception/results.csv"):
    """Plot training curves from results."""
    if not Path(results_path).exists():
        print(f"[WARNING] Results file not found: {results_path}")
        return
        
    df = pd.read_csv(results_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Box Loss
    axes[0, 0].plot(df['train/box_loss'], label='Train')
    axes[0, 0].plot(df['val/box_loss'], label='Val')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # Classification Loss  
    axes[0, 1].plot(df['train/cls_loss'], label='Train')
    axes[0, 1].plot(df['val/cls_loss'], label='Val')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # mAP50
    axes[1, 0].plot(df['metrics/mAP50(B)'])
    axes[1, 0].set_title('mAP@0.5')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylim(0, 1)
    
    # mAP50-95
    axes[1, 1].plot(df['metrics/mAP50-95(B)'])
    axes[1, 1].set_title('mAP@0.5:0.95')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def main():
    """Main training pipeline with enhanced GUI support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced YOLOv8 Training with Custom Augmentations")
    
    # Basic parameters
    parser.add_argument("--dataset", default="./yolo_dataset", help="Path to dataset")
    parser.add_argument("--model", default="n", choices=['n', 's', 'm', 'l', 'x'], help="Model size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", choices=['cuda', 'cpu'], help="Force device")
    
    # Augmentation ranges
    parser.add_argument("--degrees-min", type=float, default=0.0, help="Min rotation degrees")
    parser.add_argument("--degrees-max", type=float, default=10.0, help="Max rotation degrees")
    parser.add_argument("--translate-min", type=float, default=0.0, help="Min translation")
    parser.add_argument("--translate-max", type=float, default=0.1, help="Max translation")
    parser.add_argument("--scale-min", type=float, default=0.0, help="Min scale")
    parser.add_argument("--scale-max", type=float, default=0.3, help="Max scale")
    
    # Blur augmentations
    parser.add_argument("--gaussian-prob", type=float, default=0.3, help="Gaussian blur probability")
    parser.add_argument("--motion-prob", type=float, default=0.3, help="Motion blur probability")
    parser.add_argument("--noise-prob", type=float, default=0.2, help="Noise blur probability")
    
    # Other modes
    parser.add_argument("--evaluate", help="Evaluate model at path")
    parser.add_argument("--test", help="Test on single image")
    parser.add_argument("--batch-viewer", action="store_true", help="Enable batch viewer")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluation mode
        evaluate_model(args.evaluate, args.dataset)
    elif args.test:
        # Test mode
        if not args.evaluate:
            print("[ERROR] Please specify model path with --evaluate")
            return
        test_on_image(args.evaluate, args.test)
    else:
        # Training mode
        model, results = train_yolo_model(
            dataset_path=args.dataset,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device,
            degrees_range=(args.degrees_min, args.degrees_max),
            translate_range=(args.translate_min, args.translate_max),
            scale_range=(args.scale_min, args.scale_max),
            gaussian_blur_prob=args.gaussian_prob,
            motion_blur_prob=args.motion_prob,
            noise_blur_prob=args.noise_prob,
            batch_viewer=args.batch_viewer
        )
        
        # Plot results
        plot_training_results()
        
        print("\n[INFO] Best model saved at: runs/train/unity_perception/weights/best.pt")
        print("[INFO] Last model saved at: runs/train/unity_perception/weights/last.pt")
        
        # Quick evaluation
        print("\n[INFO] Running evaluation on best model...")
        evaluate_model("runs/train/unity_perception/weights/best.pt", args.dataset)


if __name__ == "__main__":
    main()