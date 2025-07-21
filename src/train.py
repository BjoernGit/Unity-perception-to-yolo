"""
YOLOv8 Training Script for Unity Perception Data
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import pandas as pd


def train_yolo_model(
    dataset_path="./yolo_dataset",
    model_size='n',  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    device=None
):
    """
    Train YOLOv8 model on converted dataset.
    
    Args:
        dataset_path: Path to YOLO dataset with dataset.yaml
        model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: cuda or cpu (None for auto-detect)
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
    
    # Train the model
    print(f"[INFO] Starting training for {epochs} epochs...")
    results = model.train(
        data=f'{dataset_path}/dataset.yaml',
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project='runs/train',
        name='unity_perception',
        exist_ok=True,
        
        # Optimization
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Augmentation (wichtig für kleine Datasets!)
        degrees=10.0,  # Rotation
        translate=0.1,  # Translation  
        scale=0.3,  # Scale variation
        shear=5.0,  # Shear
        perspective=0.0001,  # Perspective
        flipud=0.0,  # No vertical flip for products
        fliplr=0.5,  # Horizontal flip
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.2,  # Mixup augmentation
        copy_paste=0.1,  # Copy-paste augmentation
        
        # Hyperparameters
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss gain
        cls=0.5,  # Classification loss gain
        
        # Other settings
        workers=8,
        amp=True,  # Automatic mixed precision
        val=True,  # Validate during training
        plots=True,  # Create plots
        verbose=True
    )
    
    print("[INFO] Training completed!")
    return model, results


def evaluate_model(model_path, dataset_path="./yolo_dataset"):
    """
    Evaluate trained model on validation set.
    """
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
    """
    Test model on a single image.
    """
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
    """
    Plot training curves from results.
    """
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
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Unity Perception data")
    parser.add_argument("--dataset", default="./yolo_dataset", help="Path to dataset")
    parser.add_argument("--model", default="n", choices=['n', 's', 'm', 'l', 'x'], 
                        help="Model size (default: n)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", choices=['cuda', 'cpu'], help="Force device")
    parser.add_argument("--evaluate", help="Evaluate model at path")
    parser.add_argument("--test", help="Test on single image")
    
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
            device=args.device
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