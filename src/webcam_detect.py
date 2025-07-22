"""
Real-time Webcam Detection with YOLOv8
"""

import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import time


class WebcamDetector:
    def __init__(self, model_path='runs/train/unity_perception/weights/best.pt', 
                 camera_index=0, confidence_threshold=0.5):
        """
        Initialize webcam detector.
        
        Args:
            model_path: Path to trained YOLO model
            camera_index: Webcam index (0 for default camera)
            confidence_threshold: Minimum confidence for detections
        """
        print(f"[INFO] Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.camera_index = camera_index
        self.conf_threshold = confidence_threshold
        
        # Get class names from model
        self.class_names = self.model.names
        print(f"[INFO] Loaded model with classes: {self.class_names}")
        
        # Generate random colors for each class
        self.colors = self._generate_colors(len(self.class_names))
        
    def _generate_colors(self, num_classes):
        """Generate random colors for visualization."""
        colors = []
        np.random.seed(42)  # For consistent colors
        for i in range(num_classes):
            # Generate random BGR colors
            color = [int(x) for x in np.random.randint(0, 255, 3)]
            colors.append(color)
        return colors
    
    def draw_detections(self, image, results):
        """Draw bounding boxes and labels on image."""
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                cls = int(box.cls)
                conf = float(box.conf)
                
                if conf < self.conf_threshold:
                    continue
                
                # Get class name and color
                class_name = self.class_names[cls]
                color = self.colors[cls]
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                
                cv2.rectangle(image, 
                            (x1, label_y - label_size[1] - 4),
                            (x1 + label_size[0], label_y + 4),
                            color, -1)
                
                # Draw label text
                cv2.putText(image, label,
                          (x1, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 1)
        
        return image
    
    def run(self):
        """Run webcam detection loop."""
        # Open webcam
        print(f"[INFO] Opening camera {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print("[ERROR] Cannot open camera!")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[INFO] Starting detection... Press 'q' to quit, 's' to save screenshot")
        
        # FPS calculation
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Can't receive frame")
                break
            
            # Run detection
            results = self.model(frame, conf=self.conf_threshold)
            
            # Draw detections
            frame = self.draw_detections(frame, results)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_counter = 0
            
            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('YOLOv8 Webcam Detection', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[INFO] Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Detection stopped")


def main():
    """Main function with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8 Webcam Detection")
    parser.add_argument("--model", default="runs/train/unity_perception/weights/best.pt",
                        help="Path to YOLO model")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--test-image", help="Test on single image instead of webcam")
    
    args = parser.parse_args()
    
    if args.test_image:
        # Test mode on single image
        print(f"[INFO] Testing on image: {args.test_image}")
        model = YOLO(args.model)
        results = model(args.test_image, conf=args.conf)
        
        # Show results
        for r in results:
            im_array = r.plot()
            cv2.imshow('Detection Result', im_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Webcam mode
        detector = WebcamDetector(
            model_path=args.model,
            camera_index=args.camera,
            confidence_threshold=args.conf
        )
        detector.run()


if __name__ == "__main__":
    main()