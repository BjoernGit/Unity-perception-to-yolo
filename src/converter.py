
"""
Unity Perception to YOLO Format Converter
Converts Unity Perception synthetic dataset outputs to YOLO format for training.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


class PerceptionToYOLOConverter:
    """
    Converts Unity Perception output to YOLO format.
    
    Unity Perception format:
    - RGB images in RGB folder
    - JSON files with annotations in Dataset folder
    
    YOLO format:
    - Images in images/train and images/val
    - Labels in labels/train and labels/val
    - Each label file contains: class_id x_center y_center width height (normalized)
    """
    
    def __init__(self, perception_path: str, output_path: str):
        """
        Initialize converter.
        
        Args:
            perception_path: Path to Unity Perception output folder
            output_path: Path where YOLO dataset will be created
        """
        self.perception_path = Path(perception_path)
        self.output_path = Path(output_path)
        
        # Validate input path
        if not self.perception_path.exists():
            raise ValueError(f"Perception path does not exist: {perception_path}")
        
        # Create output structure
        self._create_output_structure()
        
    def _create_output_structure(self):
        """Create YOLO dataset folder structure."""
        # Main folders
        self.images_path = self.output_path / "images"
        self.labels_path = self.output_path / "labels"
        
        # Train/val splits
        for split in ["train", "val"]:
            (self.images_path / split).mkdir(parents=True, exist_ok=True)
            (self.labels_path / split).mkdir(parents=True, exist_ok=True)
            
    def convert(self, train_split: float = 0.8, visualize: bool = False) -> Dict:
        """
        Main conversion function.
        
        Args:
            train_split: Percentage of data for training (0.0-1.0)
            visualize: Whether to create visualization samples
            
        Returns:
            Dictionary with conversion statistics
        """
        print("🔄 Starting Unity Perception to YOLO conversion...")
        
        # Find Unity Perception files
        rgb_folder = self.perception_path / "RGB"
        dataset_folder = self.perception_path / "Dataset"
        
        if not rgb_folder.exists() or not dataset_folder.exists():
            raise ValueError("Invalid Perception dataset structure. Need RGB/ and Dataset/ folders.")
        
        # Load captures metadata
        captures_data = self._load_captures(dataset_folder)
        
        # Load annotations
        annotations, label_mapping = self._load_annotations(dataset_folder)
        
        # Convert to YOLO format
        stats = self._convert_to_yolo(
            captures_data, 
            annotations, 
            label_mapping, 
            train_split,
            visualize
        )
        
        # Create dataset.yaml
        self._create_dataset_yaml(label_mapping)
        
        # Print summary
        self._print_summary(stats, label_mapping)
        
        return stats
        
    def _load_captures(self, dataset_folder: Path) -> Dict:
        """Load captures metadata from JSON."""
        captures_files = list(dataset_folder.glob("captures_*.json"))
        if not captures_files:
            raise ValueError("No captures_*.json file found!")
            
        with open(captures_files[0], 'r') as f:
            return json.load(f)
            
    def _load_annotations(self, dataset_folder: Path) -> Tuple[Dict, Dict]:
        """Load bounding box annotations from metrics files."""
        annotations = {}
        label_mapping = {}
        
        metrics_files = list(dataset_folder.glob("metrics_*.json"))
        print(f"📁 Found {len(metrics_files)} metrics files")
        
        for metrics_file in tqdm(metrics_files, desc="Loading annotations"):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            for metric in metrics.get('metrics', []):
                # Check if this is a bounding box annotation
                if metric.get('@type') == 'type.unity.com/unity.solo.BoundingBox2DAnnotation':
                    capture_id = metric['sensorId']
                    
                    boxes = []
                    for item in metric.get('values', []):
                        label_name = item.get('labelName', f'class_{item["labelId"]}')
                        
                        # Create label mapping
                        if label_name not in label_mapping:
                            label_mapping[label_name] = len(label_mapping)
                            
                        # Store box info
                        boxes.append({
                            'class_id': label_mapping[label_name],
                            'x': item['x'],
                            'y': item['y'],
                            'width': item['width'],
                            'height': item['height']
                        })
                        
                    if boxes:  # Only store if there are boxes
                        annotations[capture_id] = boxes
                        
        return annotations, label_mapping
        
    def _convert_to_yolo(self, captures_data: Dict, annotations: Dict, 
                         label_mapping: Dict, train_split: float,
                         visualize: bool) -> Dict:
        """Convert annotations to YOLO format and copy images."""
        captures = captures_data['captures']
        total_images = len(captures)
        train_count = int(total_images * train_split)
        
        stats = {
            'total_images': total_images,
            'train_images': 0,
            'val_images': 0,
            'images_with_annotations': 0,
            'total_boxes': 0,
            'skipped_images': 0
        }
        
        print(f"📸 Processing {total_images} images...")
        
        for idx, capture in enumerate(tqdm(captures, desc="Converting")):
            capture_id = capture['id']
            filename = capture['filename']
            
            # Determine split
            split = "train" if idx < train_count else "val"
            
            # Source image path
            src_image = self.perception_path / filename
            
            if not src_image.exists():
                stats['skipped_images'] += 1
                continue
                
            # Open image to get dimensions
            try:
                img = Image.open(src_image)
                img_width, img_height = img.size
            except Exception as e:
                print(f"⚠️ Error reading image {src_image}: {e}")
                stats['skipped_images'] += 1
                continue
                
            # Copy image to destination
            dst_image = self.images_path / split / f"{capture_id}.png"
            shutil.copy2(src_image, dst_image)
            
            # Update stats
            if split == "train":
                stats['train_images'] += 1
            else:
                stats['val_images'] += 1
                
            # Create YOLO labels if annotations exist
            if capture_id in annotations:
                stats['images_with_annotations'] += 1
                yolo_labels = []
                
                for box in annotations[capture_id]:
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (box['x'] + box['width'] / 2) / img_width
                    y_center = (box['y'] + box['height'] / 2) / img_height
                    norm_width = box['width'] / img_width
                    norm_height = box['height'] / img_height
                    
                    # Ensure values are in valid range [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    norm_width = max(0, min(1, norm_width))
                    norm_height = max(0, min(1, norm_height))
                    
                    # YOLO format: class_id x_center y_center width height
                    yolo_labels.append(
                        f"{box['class_id']} {x_center:.6f} {y_center:.6f} "
                        f"{norm_width:.6f} {norm_height:.6f}"
                    )
                    stats['total_boxes'] += 1
                    
                # Save label file
                label_file = self.labels_path / split / f"{capture_id}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_labels))
            else:
                # Create empty label file for images without annotations
                label_file = self.labels_path / split / f"{capture_id}.txt"
                label_file.touch()
                
        return stats
        
    def _create_dataset_yaml(self, label_mapping: Dict):
        """Create dataset.yaml configuration file for YOLOv8."""
        # Sort labels by their ID
        sorted_labels = sorted(label_mapping.items(), key=lambda x: x[1])
        
        yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated from Unity Perception Dataset

# Path to dataset (relative to this file or absolute)
path: {self.output_path.absolute()}

# Dataset directories (relative to 'path')
train: images/train
val: images/val

# Number of classes
nc: {len(label_mapping)}

# Class names
names:"""
        
        for label_name, label_id in sorted_labels:
            yaml_content += f"\n  {label_id}: {label_name}"
            
        # Save dataset.yaml
        yaml_path = self.output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        print(f"✅ Created dataset.yaml at {yaml_path}")
        
    def _print_summary(self, stats: Dict, label_mapping: Dict):
        """Print conversion summary."""
        print("\n" + "="*50)
        print("📊 CONVERSION SUMMARY")
        print("="*50)
        print(f"Total images processed: {stats['total_images']}")
        print(f"  ├─ Training images: {stats['train_images']}")
        print(f"  └─ Validation images: {stats['val_images']}")
        print(f"Images with annotations: {stats['images_with_annotations']}")
        print(f"Total bounding boxes: {stats['total_boxes']}")
        print(f"Skipped images: {stats['skipped_images']}")
        print(f"\nClasses detected ({len(label_mapping)}):")
        for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            print(f"  {idx}: {label}")
        print("="*50)
        print(f"✅ Dataset ready at: {self.output_path}")
        print(f"🚀 You can now train with: yolo train data={self.output_path}/dataset.yaml")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Unity Perception to YOLO format")
    parser.add_argument("input", help="Path to Unity Perception output folder")
    parser.add_argument("output", help="Path for YOLO dataset output")
    parser.add_argument("--split", type=float, default=0.8, 
                        help="Train/val split ratio (default: 0.8)")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualization samples")
    
    args = parser.parse_args()
    
    # Run conversion
    converter = PerceptionToYOLOConverter(args.input, args.output)
    converter.convert(train_split=args.split, visualize=args.visualize)


if __name__ == "__main__":
    main()