"""
Flexible SOLO (Unity Perception) to YOLO Format Converter
Supports both step-based and direct file structure
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


class FlexibleSOLOToYOLOConverter:
    """
    Converts Unity Perception SOLO format to YOLO format.
    
    Supports two SOLO structures:
    1. With steps: sequence.X/stepY/stepY.camera.png
    2. Direct: sequence.X/stepY.camera.png
    """
    
    def __init__(self, solo_path: str, output_path: str):
        """Initialize converter."""
        self.solo_path = Path(solo_path)
        self.output_path = Path(output_path)
        
        if not self.solo_path.exists():
            raise ValueError(f"SOLO path does not exist: {solo_path}")
        
        self._create_output_structure()
        
    def _create_output_structure(self):
        """Create YOLO dataset folder structure."""
        self.images_path = self.output_path / "images"
        self.labels_path = self.output_path / "labels"
        
        for split in ["train", "val"]:
            (self.images_path / split).mkdir(parents=True, exist_ok=True)
            (self.labels_path / split).mkdir(parents=True, exist_ok=True)
            
    def convert(self, train_split: float = 0.8) -> Dict:
        """Main conversion function."""
        print("[Converting] Starting SOLO to YOLO conversion...")
        
        # Find all sequences
        sequences = sorted([d for d in self.solo_path.iterdir() 
                          if d.is_dir() and d.name.startswith("sequence")])
        print(f"[Folders] Found {len(sequences)} sequences")
        
        if not sequences:
            raise ValueError("No sequence folders found!")
        
        # Collect all frames
        all_frames = []
        label_mapping = {}
        
        print("[Scanning] Scanning sequences...")
        for sequence in tqdm(sequences, desc="Scanning"):
            frames = self._process_sequence(sequence, label_mapping)
            all_frames.extend(frames)
        
        print(f"[Images] Found {len(all_frames)} frames total")
        
        if not all_frames:
            print("[ERROR] No valid frames found! Check your Unity output.")
            return {'total_images': 0}
        
        # Split into train/val
        train_count = int(len(all_frames) * train_split)
        np.random.shuffle(all_frames)
        
        train_frames = all_frames[:train_count]
        val_frames = all_frames[train_count:]
        
        # Convert to YOLO format
        stats = {
            'total_images': len(all_frames),
            'train_images': len(train_frames),
            'val_images': len(val_frames),
            'images_with_annotations': 0,
            'total_boxes': 0,
            'skipped_images': 0
        }
        
        print("[Converting] Converting to YOLO format...")
        self._convert_frames(train_frames, "train", stats, label_mapping)
        self._convert_frames(val_frames, "val", stats, label_mapping)
        
        # Create dataset.yaml
        if label_mapping:
            self._create_dataset_yaml(label_mapping)
        
        # Print summary
        self._print_summary(stats, label_mapping)
        
        return stats
        
    def _process_sequence(self, sequence_path: Path, label_mapping: Dict) -> List:
        """Process a single sequence folder - handles both formats."""
        frames = []
        
        # Check for direct files (format 2)
        direct_images = list(sequence_path.glob("step*.camera.png"))
        if direct_images:
            print(f"  Found {len(direct_images)} images in {sequence_path.name} (direct format)")
            for image_file in direct_images:
                # Find corresponding JSON
                json_name = image_file.name.replace('.camera.png', '.frame_data.json')
                json_file = sequence_path / json_name
                
                if json_file.exists():
                    frames.extend(self._process_frame(image_file, json_file, label_mapping))
        
        # Check for step folders (format 1)
        else:
            steps = sorted([d for d in sequence_path.iterdir() 
                          if d.is_dir() and d.name.startswith("step")])
            
            if steps:
                print(f"  Found {len(steps)} steps in {sequence_path.name} (step format)")
                for step in steps:
                    image_file = step / f"{step.name}.camera.png"
                    json_file = step / f"{step.name}.frame_data.json"
                    
                    if image_file.exists() and json_file.exists():
                        frames.extend(self._process_frame(image_file, json_file, label_mapping))
        
        return frames
        
    def _process_frame(self, image_file: Path, json_file: Path, label_mapping: Dict) -> List:
        """Process a single frame (image + json)."""
        frames = []
        
        try:
            with open(json_file, 'r') as f:
                frame_data = json.load(f)
            
            # Extract bounding boxes
            for capture in frame_data.get('captures', []):
                dimension = capture.get('dimension', [640, 480])  # Default size
                
                for annotation in capture.get('annotations', []):
                    if annotation.get('@type') == 'type.unity.com/unity.solo.BoundingBox2DAnnotation':
                        # Process labels
                        for value in annotation.get('values', []):
                            label_name = value.get('labelName', 'unknown')
                            if label_name not in label_mapping:
                                label_mapping[label_name] = len(label_mapping)
                        
                        # Store frame info
                        frames.append({
                            'image_path': image_file,
                            'annotation': annotation,
                            'dimension': dimension
                        })
                        break  # Only take first bbox annotation
                        
        except Exception as e:
            print(f"[Warning] Error reading {json_file}: {e}")
            
        return frames
        
    def _convert_frames(self, frames: List, split: str, stats: Dict, label_mapping: Dict):
        """Convert frames to YOLO format."""
        for frame in tqdm(frames, desc=f"Converting {split}"):
            try:
                # Copy image
                src_image = frame['image_path']
                # Create unique name
                seq_name = src_image.parent.name if "step" not in src_image.parent.name else src_image.parent.parent.name
                image_name = f"{seq_name}_{src_image.stem}.png"
                dst_image = self.images_path / split / image_name
                shutil.copy2(src_image, dst_image)
                
                # Get image dimensions
                img_width, img_height = frame['dimension']
                
                # Create YOLO labels
                annotation = frame['annotation']
                yolo_labels = []
                
                values = annotation.get('values', [])
                if values:
                    stats['images_with_annotations'] += 1
                    
                    for box in values:
                        # Get label ID
                        label_name = box.get('labelName', 'unknown')
                        class_id = label_mapping.get(label_name, 0)
                        
                        # Convert origin/dimension to YOLO format
                        origin = box.get('origin', [0, 0])
                        dimension = box.get('dimension', [0, 0])
                        
                        x_origin, y_origin = origin
                        width, height = dimension
                        
                        # Skip invalid boxes
                        if width <= 0 or height <= 0:
                            continue
                        
                        # Calculate center coordinates
                        x_center = (x_origin + width / 2) / img_width
                        y_center = (y_origin + height / 2) / img_height
                        norm_width = width / img_width
                        norm_height = height / img_height
                        
                        # Ensure values are in valid range [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        norm_width = max(0, min(1, norm_width))
                        norm_height = max(0, min(1, norm_height))
                        
                        yolo_labels.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} "
                            f"{norm_width:.6f} {norm_height:.6f}"
                        )
                        stats['total_boxes'] += 1
                
                # Save label file
                label_name = image_name.replace('.png', '.txt')
                label_file = self.labels_path / split / label_name
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                    
            except Exception as e:
                print(f"[Warning] Error processing {frame.get('image_path', 'unknown')}: {e}")
                stats['skipped_images'] += 1
                
    def _create_dataset_yaml(self, label_mapping: Dict):
        """Create dataset.yaml configuration file for YOLOv8."""
        sorted_labels = sorted(label_mapping.items(), key=lambda x: x[1])
        
        yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated from Unity Perception SOLO Dataset

# Path to dataset
path: {self.output_path.absolute()}

# Dataset directories
train: images/train
val: images/val

# Number of classes
nc: {len(label_mapping)}

# Class names
names:"""
        
        for label_name, label_id in sorted_labels:
            yaml_content += f"\n  {label_id}: {label_name}"
            
        yaml_path = self.output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        print(f"[OK] Created dataset.yaml at {yaml_path}")
        
    def _print_summary(self, stats: Dict, label_mapping: Dict):
        """Print conversion summary."""
        print("\n" + "="*50)
        print("[SUMMARY] CONVERSION SUMMARY")
        print("="*50)
        print(f"Total images processed: {stats['total_images']}")
        print(f"  |- Training images: {stats['train_images']}")
        print(f"  '- Validation images: {stats['val_images']}")
        print(f"Images with annotations: {stats['images_with_annotations']}")
        print(f"Total bounding boxes: {stats['total_boxes']}")
        print(f"Skipped images: {stats['skipped_images']}")
        
        if label_mapping:
            print(f"\nClasses detected ({len(label_mapping)}):")
            for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
                print(f"  {idx}: {label}")
        else:
            print("\n[WARNING] No classes detected!")
            
        print("="*50)
        
        if stats['total_images'] > 0:
            print(f"[OK] Dataset ready at: {self.output_path}")
            print(f"[Ready] You can now train with: yolo train data={self.output_path}/dataset.yaml")
        else:
            print("[ERROR] No images were converted. Check your Unity output!")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Unity Perception SOLO to YOLO format")
    parser.add_argument("input", help="Path to SOLO dataset folder")
    parser.add_argument("output", help="Path for YOLO dataset output")
    parser.add_argument("--split", type=float, default=0.8, 
                        help="Train/val split ratio (default: 0.8)")
    
    args = parser.parse_args()
    
    converter = FlexibleSOLOToYOLOConverter(args.input, args.output)
    converter.convert(train_split=args.split)


if __name__ == "__main__":
    main()