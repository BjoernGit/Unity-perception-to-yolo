"""
YOLOv8 Training GUI
Interactive interface for training YOLO models with Unity Perception data
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import sys
import os
from pathlib import Path
import yaml
import time


class YOLOTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Training GUI")
        self.root.geometry("800x900")
        
        # Training process
        self.training_process = None
        self.is_training = False
        
        # Variables
        self.setup_variables()
        self.setup_ui()
        
    def setup_variables(self):
        """Initialize all GUI variables with defaults."""
        # Dataset settings
        self.dataset_path = tk.StringVar(value="./yolo_dataset")
        self.output_project = tk.StringVar(value="runs/train")
        self.output_name = tk.StringVar(value="unity_perception")
        
        # Model settings  
        self.model_size = tk.StringVar(value="m")
        self.pretrained_model = tk.StringVar(value="")  # Empty = use default yolov8{size}.pt
        
        # Training parameters
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=16)
        self.image_size = tk.IntVar(value=640)
        self.device = tk.StringVar(value="auto")
        self.workers = tk.IntVar(value=8)
        
        # Optimization
        self.patience = tk.IntVar(value=20)
        self.save_period = tk.IntVar(value=10)
        self.lr0 = tk.DoubleVar(value=0.01)
        self.lrf = tk.DoubleVar(value=0.01)
        self.momentum = tk.DoubleVar(value=0.937)
        self.weight_decay = tk.DoubleVar(value=0.0005)
        
        # Standard augmentation parameters (ranges)
        self.degrees_min = tk.DoubleVar(value=0.0)
        self.degrees_max = tk.DoubleVar(value=10.0)
        self.translate_min = tk.DoubleVar(value=0.0)
        self.translate_max = tk.DoubleVar(value=0.1)
        self.scale_min = tk.DoubleVar(value=0.0)
        self.scale_max = tk.DoubleVar(value=0.3)
        self.shear_min = tk.DoubleVar(value=0.0)
        self.shear_max = tk.DoubleVar(value=5.0)
        self.perspective_min = tk.DoubleVar(value=0.0)
        self.perspective_max = tk.DoubleVar(value=0.0001)
        
        # Flip probabilities
        self.flipud_prob = tk.DoubleVar(value=0.0)  # Vertical flip probability
        self.fliplr_prob = tk.DoubleVar(value=0.5)  # Horizontal flip probability
        
        # Advanced augmentation probabilities
        self.mosaic_prob = tk.DoubleVar(value=1.0)
        self.mixup_prob = tk.DoubleVar(value=0.2)
        self.copy_paste_prob = tk.DoubleVar(value=0.1)
        
        # Custom blur augmentations (ranges)
        self.gaussian_blur_prob = tk.DoubleVar(value=0.3)
        self.gaussian_kernel_min = tk.IntVar(value=3)
        self.gaussian_kernel_max = tk.IntVar(value=15)
        self.gaussian_sigma_min = tk.DoubleVar(value=1.0)
        self.gaussian_sigma_max = tk.DoubleVar(value=5.0)
        
        self.motion_blur_prob = tk.DoubleVar(value=0.3)
        self.motion_length_min = tk.IntVar(value=5)
        self.motion_length_max = tk.IntVar(value=25)
        self.motion_angle_min = tk.DoubleVar(value=0.0)
        self.motion_angle_max = tk.DoubleVar(value=180.0)
        
        self.radial_blur_prob = tk.DoubleVar(value=0.1)
        self.radial_strength_min = tk.IntVar(value=1)
        self.radial_strength_max = tk.IntVar(value=15)
        
        self.noise_blur_prob = tk.DoubleVar(value=0.2)
        self.noise_strength_min = tk.IntVar(value=5)
        self.noise_strength_max = tk.IntVar(value=30)
        
        # Loss weights
        self.box_loss = tk.DoubleVar(value=7.5)
        self.cls_loss = tk.DoubleVar(value=0.5)
        
        # Advanced options
        self.amp = tk.BooleanVar(value=True)
        self.val = tk.BooleanVar(value=True)
        self.plots = tk.BooleanVar(value=True)
        self.exist_ok = tk.BooleanVar(value=True)
        
        # Augmentation preview
        self.preview_window = None
        self.preview_images = []
        self.grid_size = tk.StringVar(value="2x2")  # Default grid size
        
    def setup_ui(self):
        """Setup the user interface."""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main settings tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Main Settings")
        self.setup_main_tab(main_frame)
        
        # Augmentation tab
        aug_frame = ttk.Frame(notebook)
        notebook.add(aug_frame, text="Augmentation")
        self.setup_augmentation_tab(aug_frame)
        
        # Advanced tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        self.setup_advanced_tab(advanced_frame)
        
        # Training tab
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="Training")
        self.setup_training_tab(training_frame)
        
    def setup_main_tab(self, parent):
        """Setup main settings tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Dataset Settings
        dataset_frame = ttk.LabelFrame(scrollable_frame, text="Dataset Settings", padding="10")
        dataset_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(dataset_frame, text="Dataset Path:").grid(row=0, column=0, sticky=tk.W)
        dataset_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=40)
        dataset_entry.grid(row=0, column=1, padx=(5, 5), sticky=(tk.W, tk.E))
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2)
        
        ttk.Label(dataset_frame, text="Output Project:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(dataset_frame, textvariable=self.output_project, width=40).grid(row=1, column=1, padx=(5, 5), pady=(5, 0), sticky=(tk.W, tk.E))
        
        ttk.Label(dataset_frame, text="Output Name:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(dataset_frame, textvariable=self.output_name, width=40).grid(row=2, column=1, padx=(5, 5), pady=(5, 0), sticky=(tk.W, tk.E))
        
        # Model Settings
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Settings", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="Model Size:").grid(row=0, column=0, sticky=tk.W)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_size, 
                                  values=["n", "s", "m", "l", "x"], state="readonly", width=10)
        model_combo.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)
        
        # Model size info
        size_info = {
            "n": "Nano (3.2M params) - Fastest",
            "s": "Small (11.2M params) - Fast", 
            "m": "Medium (25.9M params) - Balanced",
            "l": "Large (43.7M params) - Accurate",
            "x": "XLarge (68.2M params) - Most Accurate"
        }
        
        self.model_info = ttk.Label(model_frame, text=size_info[self.model_size.get()])
        self.model_info.grid(row=0, column=2, padx=(10, 0), sticky=tk.W)
        
        model_combo.bind("<<ComboboxSelected>>", self.update_model_info)
        
        ttk.Label(model_frame, text="Custom Model Path:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        model_entry = ttk.Entry(model_frame, textvariable=self.pretrained_model, width=40)
        model_entry.grid(row=1, column=1, padx=(5, 5), pady=(5, 0), sticky=(tk.W, tk.E))
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=1, column=2, pady=(5, 0))
        
        # Training Parameters
        train_frame = ttk.LabelFrame(scrollable_frame, text="Training Parameters", padding="10")
        train_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create 2-column layout for parameters
        params = [
            ("Epochs:", self.epochs, 1, 1000),
            ("Batch Size:", self.batch_size, 1, 128),
            ("Image Size:", self.image_size, 320, 1280),
            ("Workers:", self.workers, 0, 16),
            ("Patience:", self.patience, 5, 100),
            ("Save Period:", self.save_period, 1, 50)
        ]
        
        for i, (label, var, min_val, max_val) in enumerate(params):
            row = i // 2
            col = (i % 2) * 3
            
            ttk.Label(train_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 5))
            spinbox = tk.Spinbox(train_frame, from_=min_val, to=max_val, textvariable=var, width=10)
            spinbox.grid(row=row, column=col+1, padx=(0, 15))
        
        # Device selection
        ttk.Label(train_frame, text="Device:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        device_combo = ttk.Combobox(train_frame, textvariable=self.device,
                                   values=["auto", "cuda", "cpu", "0", "1"], width=10)
        device_combo.grid(row=3, column=1, pady=(10, 0), sticky=tk.W)
        
        # Configure grid weights
        dataset_frame.columnconfigure(1, weight=1)
        model_frame.columnconfigure(1, weight=1)
        scrollable_frame.columnconfigure(0, weight=1)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_augmentation_tab(self, parent):
        """Setup augmentation parameters tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Geometric Augmentations (Ranges)
        geo_frame = ttk.LabelFrame(scrollable_frame, text="Geometric Augmentations (Min-Max Ranges)", padding="10")
        geo_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Helper function for range sliders
        def create_range_slider(parent, row, label, min_var, max_var, min_val, max_val, step=0.01, tooltip=""):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
            
            # Min slider
            ttk.Label(parent, text="Min:", font=("TkDefaultFont", 8)).grid(row=row, column=1, sticky=tk.W)
            min_scale = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                                 variable=min_var, length=100)
            min_scale.grid(row=row, column=2, padx=(0, 5), sticky=(tk.W, tk.E))
            
            min_label = ttk.Label(parent, text=f"{min_var.get():.3f}")
            min_label.grid(row=row, column=3, padx=(0, 10))
            
            # Max slider  
            ttk.Label(parent, text="Max:", font=("TkDefaultFont", 8)).grid(row=row, column=4, sticky=tk.W)
            max_scale = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                                 variable=max_var, length=100)
            max_scale.grid(row=row, column=5, padx=(0, 5), sticky=(tk.W, tk.E))
            
            max_label = ttk.Label(parent, text=f"{max_var.get():.3f}")
            max_label.grid(row=row, column=6)
            
            # Update labels when values change
            min_var.trace_add("write", lambda *args: min_label.config(text=f"{min_var.get():.3f}"))
            max_var.trace_add("write", lambda *args: max_label.config(text=f"{max_var.get():.3f}"))
            
            # Add tooltip as info label
            if tooltip:
                info_label = ttk.Label(parent, text=f"({tooltip})", font=("TkDefaultFont", 7), foreground="gray")
                info_label.grid(row=row, column=7, padx=(5, 0), sticky=tk.W)
        
        # Geometric augmentation ranges
        create_range_slider(geo_frame, 0, "Rotation (degrees):", self.degrees_min, self.degrees_max, 
                           0.0, 90.0, tooltip="Random rotation from min to max degrees")
        create_range_slider(geo_frame, 1, "Translation:", self.translate_min, self.translate_max,
                           0.0, 0.5, tooltip="Translation fraction of image size")
        create_range_slider(geo_frame, 2, "Scale:", self.scale_min, self.scale_max,
                           0.0, 1.0, tooltip="Scale variation factor")
        create_range_slider(geo_frame, 3, "Shear (degrees):", self.shear_min, self.shear_max,
                           0.0, 45.0, tooltip="Shear angle range")
        create_range_slider(geo_frame, 4, "Perspective:", self.perspective_min, self.perspective_max,
                           0.0, 0.01, tooltip="Perspective transform strength")
        
        # Flip Augmentations (Probabilities)
        flip_frame = ttk.LabelFrame(scrollable_frame, text="Flip Augmentations (Probabilities)", padding="10")
        flip_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        def create_probability_slider(parent, row, label, var, tooltip=""):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
            
            scale = ttk.Scale(parent, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                             variable=var, length=200)
            scale.grid(row=row, column=1, padx=(0, 10), sticky=(tk.W, tk.E))
            
            value_label = ttk.Label(parent, text=f"{var.get():.2f} ({var.get()*100:.0f}%)")
            value_label.grid(row=row, column=2, padx=(0, 5))
            
            if tooltip:
                info_label = ttk.Label(parent, text=f"({tooltip})", font=("TkDefaultFont", 7), foreground="gray")
                info_label.grid(row=row, column=3, padx=(5, 0), sticky=tk.W)
            
            # Update label with percentage
            var.trace_add("write", lambda *args: value_label.config(text=f"{var.get():.2f} ({var.get()*100:.0f}%)"))
        
        create_probability_slider(flip_frame, 0, "Horizontal Flip:", self.fliplr_prob, 
                                "Probability of horizontal flip")
        create_probability_slider(flip_frame, 1, "Vertical Flip:", self.flipud_prob,
                                "Probability of vertical flip")
        
        # Advanced Augmentations
        advanced_aug_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Augmentations (Probabilities)", padding="10")
        advanced_aug_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        create_probability_slider(advanced_aug_frame, 0, "Mosaic:", self.mosaic_prob,
                                "4-image mosaic augmentation")
        create_probability_slider(advanced_aug_frame, 1, "Mixup:", self.mixup_prob,
                                "Blend two images together")
        create_probability_slider(advanced_aug_frame, 2, "Copy-Paste:", self.copy_paste_prob,
                                "Copy objects to other images")
        
        # Custom Blur Augmentations
        blur_frame = ttk.LabelFrame(scrollable_frame, text="Custom Blur Augmentations", padding="10")
        blur_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Gaussian Blur
        gaussian_subframe = ttk.LabelFrame(blur_frame, text="Gaussian Blur", padding="5")
        gaussian_subframe.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        create_probability_slider(gaussian_subframe, 0, "Probability:", self.gaussian_blur_prob,
                                "Chance to apply Gaussian blur")
        
        def create_int_range_slider(parent, row, label, min_var, max_var, min_val, max_val, tooltip=""):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
            
            ttk.Label(parent, text="Min:", font=("TkDefaultFont", 8)).grid(row=row, column=1, sticky=tk.W)
            min_scale = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                                 variable=min_var, length=80)
            min_scale.grid(row=row, column=2, padx=(0, 5))
            
            min_label = ttk.Label(parent, text=f"{min_var.get()}")
            min_label.grid(row=row, column=3, padx=(0, 10))
            
            ttk.Label(parent, text="Max:", font=("TkDefaultFont", 8)).grid(row=row, column=4, sticky=tk.W)
            max_scale = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                                 variable=max_var, length=80)
            max_scale.grid(row=row, column=5, padx=(0, 5))
            
            max_label = ttk.Label(parent, text=f"{max_var.get()}")
            max_label.grid(row=row, column=6)
            
            min_var.trace_add("write", lambda *args: min_label.config(text=f"{int(min_var.get())}"))
            max_var.trace_add("write", lambda *args: max_label.config(text=f"{int(max_var.get())}"))
        
        create_int_range_slider(gaussian_subframe, 1, "Kernel Size:", self.gaussian_kernel_min, self.gaussian_kernel_max,
                               3, 51, "Blur kernel size (odd numbers)")
        create_range_slider(gaussian_subframe, 2, "Sigma:", self.gaussian_sigma_min, self.gaussian_sigma_max,
                           0.1, 10.0, tooltip="Gaussian blur strength")
        
        # Motion Blur
        motion_subframe = ttk.LabelFrame(blur_frame, text="Motion Blur", padding="5")
        motion_subframe.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        create_probability_slider(motion_subframe, 0, "Probability:", self.motion_blur_prob,
                                "Chance to apply motion blur")
        create_int_range_slider(motion_subframe, 1, "Length:", self.motion_length_min, self.motion_length_max,
                               3, 50, "Motion blur length")  
        create_range_slider(motion_subframe, 2, "Angle (degrees):", self.motion_angle_min, self.motion_angle_max,
                           0.0, 180.0, tooltip="Motion direction angle")
        
        # Radial Blur
        radial_subframe = ttk.LabelFrame(blur_frame, text="Radial Blur", padding="5")
        radial_subframe.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        create_probability_slider(radial_subframe, 0, "Probability:", self.radial_blur_prob,
                                "Chance to apply radial blur")
        create_int_range_slider(radial_subframe, 1, "Strength:", self.radial_strength_min, self.radial_strength_max,
                               1, 30, "Radial blur intensity")
        
        # Noise Blur
        noise_subframe = ttk.LabelFrame(blur_frame, text="Noise + Blur", padding="5")
        noise_subframe.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        create_probability_slider(noise_subframe, 0, "Probability:", self.noise_blur_prob,
                                "Chance to apply noise + blur")
        create_int_range_slider(noise_subframe, 1, "Noise Strength:", self.noise_strength_min, self.noise_strength_max,
                               0, 100, "Gaussian noise intensity")
        
        # Augmentation Preview Tool
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Augmentation Preview Tool", padding="10")
        preview_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        preview_info = tk.Text(preview_frame, height=2, width=60, wrap=tk.WORD)
        preview_info.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        preview_info.insert("1.0", "Test your augmentation settings before training! Load sample images from your dataset and see the effects live.")
        preview_info.config(state="disabled")
        
        ttk.Button(preview_frame, text="Open Augmentation Preview", 
                  command=self.open_augmentation_preview).grid(row=1, column=0, sticky=tk.W)
        
        preview_frame.columnconfigure(0, weight=1)
        
        # Configure grid weights
        geo_frame.columnconfigure(2, weight=1)
        geo_frame.columnconfigure(5, weight=1)
        flip_frame.columnconfigure(1, weight=1)
        advanced_aug_frame.columnconfigure(1, weight=1)
        blur_frame.columnconfigure(0, weight=1)
        gaussian_subframe.columnconfigure(0, weight=1)
        motion_subframe.columnconfigure(0, weight=1)
        radial_subframe.columnconfigure(0, weight=1)
        noise_subframe.columnconfigure(0, weight=1)
        scrollable_frame.columnconfigure(0, weight=1)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_advanced_tab(self, parent):
        """Setup advanced parameters tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Optimizer Settings
        opt_frame = ttk.LabelFrame(scrollable_frame, text="Optimizer Settings", padding="10")
        opt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        opt_params = [
            ("Initial Learning Rate:", self.lr0, 0.001, 0.1),
            ("Final LR Factor:", self.lrf, 0.001, 0.1),
            ("Momentum:", self.momentum, 0.8, 0.99),
            ("Weight Decay:", self.weight_decay, 0.0, 0.01)
        ]
        
        for i, (label, var, min_val, max_val) in enumerate(opt_params):
            ttk.Label(opt_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=(0, 5))
            
            scale = ttk.Scale(opt_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                             variable=var, length=200)
            scale.grid(row=i, column=1, padx=(0, 10), sticky=(tk.W, tk.E))
            
            value_label = ttk.Label(opt_frame, text=f"{var.get():.4f}")
            value_label.grid(row=i, column=2, padx=(0, 5))
            
            var.trace_add("write", lambda *args, label=value_label, v=var: 
                         label.config(text=f"{v.get():.4f}"))
        
        # Loss Weights
        loss_frame = ttk.LabelFrame(scrollable_frame, text="Loss Weights", padding="10")
        loss_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        loss_params = [
            ("Box Loss Weight:", self.box_loss, 1.0, 20.0),
            ("Classification Loss Weight:", self.cls_loss, 0.1, 2.0)
        ]
        
        for i, (label, var, min_val, max_val) in enumerate(loss_params):
            ttk.Label(loss_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=(0, 5))
            
            scale = ttk.Scale(loss_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                             variable=var, length=200)
            scale.grid(row=i, column=1, padx=(0, 10), sticky=(tk.W, tk.E))
            
            value_label = ttk.Label(loss_frame, text=f"{var.get():.1f}")
            value_label.grid(row=i, column=2, padx=(0, 5))
            
            var.trace_add("write", lambda *args, label=value_label, v=var: 
                         label.config(text=f"{v.get():.1f}"))
        
        # Advanced Options
        options_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Options", padding="10")
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text="Automatic Mixed Precision (AMP)", 
                       variable=self.amp).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Validation during training", 
                       variable=self.val).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Generate training plots", 
                       variable=self.plots).grid(row=2, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Overwrite existing runs", 
                       variable=self.exist_ok).grid(row=3, column=0, sticky=tk.W)
        
        # Configure grid weights
        opt_frame.columnconfigure(1, weight=1)
        loss_frame.columnconfigure(1, weight=1)
        scrollable_frame.columnconfigure(0, weight=1)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_training_tab(self, parent):
        """Setup training control and output tab."""
        # Control buttons
        control_frame = ttk.LabelFrame(parent, text="Training Control", padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # Training command preview
        ttk.Label(control_frame, text="Training Command:").grid(row=0, column=0, sticky=tk.W)
        self.command_text = scrolledtext.ScrolledText(control_frame, height=4, width=80, wrap=tk.WORD)
        self.command_text.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 10))
        
        # Update command preview
        self.update_command_preview()
        
        # Buttons
        self.start_button = ttk.Button(control_frame, text="Start Training", 
                                      command=self.start_training, style="Success.TButton")
        self.start_button.grid(row=2, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Training", 
                                     command=self.stop_training, state="disabled")
        self.stop_button.grid(row=2, column=1, padx=5)
        
        ttk.Button(control_frame, text="Update Preview", 
                  command=self.update_command_preview).grid(row=2, column=2, padx=(5, 0))
        
        # Progress and status
        status_frame = ttk.LabelFrame(parent, text="Training Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Status info
        self.status_label = ttk.Label(status_frame, text="Ready to train", foreground="green")
        self.status_label.pack(anchor=tk.W)
        
        # Output text area
        ttk.Label(status_frame, text="Training Output:").pack(anchor=tk.W, pady=(10, 0))
        self.output_text = scrolledtext.ScrolledText(status_frame, height=20, width=80)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Configure grid weights
        control_frame.columnconfigure(0, weight=1)
        
    def open_augmentation_preview(self):
        """Open the augmentation preview window."""
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.create_preview_window()
        else:
            self.preview_window.lift()
    
    def create_preview_window(self):
        """Create the augmentation preview window."""
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("Augmentation Preview Tool")
        self.preview_window.geometry("900x700")
        
        # Main frame
        main_frame = ttk.Frame(self.preview_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Dataset path
        ttk.Label(control_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W)
        dataset_label = ttk.Label(control_frame, text=self.dataset_path.get(), foreground="blue")
        dataset_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        # Grid size selector
        ttk.Label(control_frame, text="Grid Size:").grid(row=0, column=2, sticky=tk.W)
        grid_combo = ttk.Combobox(control_frame, textvariable=self.grid_size, 
                                 values=["1x1", "2x2", "3x3", "4x4"], state="readonly", width=8)
        grid_combo.grid(row=0, column=3, padx=(5, 20))
        grid_combo.bind("<<ComboboxSelected>>", self.on_grid_size_change)
        
        # Buttons
        ttk.Button(control_frame, text="Load Sample Images", 
                  command=self.load_sample_images).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Refresh Augmentations", 
                  command=self.refresh_augmentations).grid(row=0, column=5, padx=5)
        
        # Settings display
        settings_label = ttk.Label(control_frame, text="Current settings will be applied when refreshing", 
                                 foreground="gray", font=("TkDefaultFont", 8))
        settings_label.grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(5, 0))
        
        # Image grid frame
        self.grid_frame = ttk.LabelFrame(main_frame, text="Augmentation Preview", padding="10")
        self.grid_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create initial grid
        self.create_image_grid()
        
        # Status bar
        self.preview_status = ttk.Label(main_frame, text="Ready - Load sample images to start", 
                                      foreground="green")
        self.preview_status.pack(pady=(10, 0))
        
        # Handle window close
        self.preview_window.protocol("WM_DELETE_WINDOW", self.close_preview_window)
    
    def create_image_grid(self):
        """Create the image grid based on current grid size."""
        # Clear existing grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        
        # Get grid dimensions
        grid_sizes = {"1x1": (1, 1), "2x2": (2, 2), "3x3": (3, 3), "4x4": (4, 4)}
        rows, cols = grid_sizes[self.grid_size.get()]
        
        # Update frame title
        self.grid_frame.config(text=f"Augmentation Preview ({self.grid_size.get()} Grid)")
        
        # Create new grid
        self.preview_image_labels = []
        for row in range(rows):
            label_row = []
            for col in range(cols):
                img_label = tk.Label(self.grid_frame, 
                                   text=f"Image {row*cols+col+1}\nClick 'Load Sample Images'",
                                   relief="sunken", borderwidth=1,
                                   bg="lightgray")
                img_label.grid(row=row, column=col, padx=2, pady=2, sticky=(tk.W, tk.E, tk.N, tk.S))
                label_row.append(img_label)
            
            # Configure row weights for responsive scaling
            self.grid_frame.rowconfigure(row, weight=1)
            self.preview_image_labels.append(label_row)
        
        # Configure column weights for responsive scaling
        for col in range(cols):
            self.grid_frame.columnconfigure(col, weight=1)
        
        # Bind resize event to update images (bind to window, not frame)
        self.preview_window.bind("<Configure>", self.on_preview_resize)
    
    def on_grid_size_change(self, event=None):
        """Handle grid size change."""
        self.create_image_grid()
        
        # If images are loaded, refresh them for new grid
        if hasattr(self, 'preview_images') and self.preview_images:
            self.load_sample_images()  # Reload to fit new grid size
    
    def close_preview_window(self):
        """Close the preview window."""
        if self.preview_window and self.preview_window.winfo_exists():
            self.preview_window.destroy()
        self.preview_window = None
    
    def load_sample_images(self):
        """Load sample images from the dataset."""
        import random
        import cv2
        
        try:
            dataset_path = Path(self.dataset_path.get())
            if not dataset_path.exists():
                messagebox.showerror("Error", "Dataset path does not exist!")
                return
            
            # Look for images in train folder
            train_images_path = dataset_path / "images" / "train"
            if not train_images_path.exists():
                messagebox.showerror("Error", "Train images folder not found!")
                return
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(train_images_path.glob(ext))
            
            if not image_files:
                messagebox.showerror("Error", "No images found in train folder!")
                return
            
            # Select images based on grid size
            import random
            grid_sizes = {"1x1": 1, "2x2": 4, "3x3": 9, "4x4": 16}
            max_images = grid_sizes[self.grid_size.get()]
            selected_images = random.sample(image_files, min(max_images, len(image_files)))
            
            # Load and display images
            self.preview_images = []
            for i, img_path in enumerate(selected_images):
                if i >= max_images:
                    break
                    
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.preview_images.append(img.copy())
                        
                        # Display original image
                        self.display_preview_image(img, i, "Original")
                    
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            
            # Fill empty slots if less than max images
            while len(self.preview_images) < max_images:
                if self.preview_images:
                    self.preview_images.append(self.preview_images[0].copy())
                else:
                    break
            
            self.preview_status.config(text=f"Loaded {len(selected_images)} sample images", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample images: {str(e)}")
    
    def refresh_augmentations(self):
        """Apply current augmentation settings to preview images."""
        if not self.preview_images:
            messagebox.showwarning("Warning", "Please load sample images first!")
            return
        
        try:
            import cv2
            import numpy as np
            import random
            
            # Get max images for current grid
            grid_sizes = {"1x1": 1, "2x2": 4, "3x3": 9, "4x4": 16}
            max_images = grid_sizes[self.grid_size.get()]
            
            # Apply augmentations to each image
            for i, original_img in enumerate(self.preview_images[:max_images]):
                
                # Start with original image
                augmented_img = original_img.copy()
                applied_effects = []
                
                # Apply Geometric Augmentations
                # Rotation
                if self.degrees_max.get() > 0:
                    angle = random.uniform(self.degrees_min.get(), self.degrees_max.get())
                    if abs(angle) > 0.1:  # Only rotate if significant
                        h, w = augmented_img.shape[:2]
                        center = (w // 2, h // 2)
                        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        augmented_img = cv2.warpAffine(augmented_img, matrix, (w, h))
                        applied_effects.append(f"Rot:{angle:.1f}°")
                
                # Horizontal Flip
                if random.random() < self.fliplr_prob.get():
                    augmented_img = cv2.flip(augmented_img, 1)
                    applied_effects.append("HFlip")
                
                # Gaussian Blur
                if random.random() < self.gaussian_blur_prob.get():
                    kernel_size = random.randint(self.gaussian_kernel_min.get(), self.gaussian_kernel_max.get())
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    sigma = random.uniform(self.gaussian_sigma_min.get(), self.gaussian_sigma_max.get())
                    augmented_img = cv2.GaussianBlur(augmented_img, (kernel_size, kernel_size), sigma)
                    applied_effects.append(f"Gauss({kernel_size},{sigma:.1f})")
                
                # Motion Blur
                if random.random() < self.motion_blur_prob.get():
                    length = random.randint(self.motion_length_min.get(), self.motion_length_max.get())
                    angle = random.uniform(self.motion_angle_min.get(), self.motion_angle_max.get())
                    
                    # Create motion blur kernel
                    kernel = np.zeros((length, length))
                    center = length // 2
                    angle_rad = np.radians(angle)
                    
                    for j in range(length):
                        offset = j - center
                        x = int(center + offset * np.cos(angle_rad))
                        y = int(center + offset * np.sin(angle_rad))
                        if 0 <= x < length and 0 <= y < length:
                            kernel[y, x] = 1
                    
                    if np.sum(kernel) > 0:
                        kernel = kernel / np.sum(kernel)
                        augmented_img = cv2.filter2D(augmented_img, -1, kernel)
                        applied_effects.append(f"Motion({length},{angle:.0f}°)")
                
                # Noise Blur
                if random.random() < self.noise_blur_prob.get():
                    noise_strength = random.randint(self.noise_strength_min.get(), self.noise_strength_max.get())
                    noise = np.random.normal(0, noise_strength, augmented_img.shape).astype(np.int16)
                    noisy = np.clip(augmented_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    augmented_img = cv2.GaussianBlur(noisy, (3, 3), 1.0)
                    applied_effects.append(f"Noise({noise_strength})")
                
                # Create title
                effects_title = " + ".join(applied_effects) if applied_effects else "No effects"
                
                # Display augmented image
                self.display_preview_image(augmented_img, i, effects_title)
            
            self.preview_status.config(text="Augmentations applied successfully", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply augmentations: {str(e)}")
            self.preview_status.config(text=f"Error: {str(e)}", foreground="red")
    
    def display_preview_image(self, image, index, title):
        """Display an image in the preview grid."""
        try:
            from PIL import Image, ImageTk
            
            # Get grid dimensions
            grid_sizes = {"1x1": (1, 1), "2x2": (2, 2), "3x3": (3, 3), "4x4": (4, 4)}
            rows, cols = grid_sizes[self.grid_size.get()]
            max_images = rows * cols
            
            if index >= max_images:
                return
            
            row = index // cols
            col = index % cols
            
            # Get current label size for responsive scaling
            label = self.preview_image_labels[row][col]
            
            # Force geometry update
            label.update_idletasks()
            
            # Get available space (with some padding)
            label_width = label.winfo_width()
            label_height = label.winfo_height()
            
            # Only calculate size if widget is properly initialized
            if label_width > 1 and label_height > 1:
                available_width = max(100, label_width - 10)
                available_height = max(100, label_height - 10)
                
                # Use minimum of width/height to keep square aspect ratio
                size = min(available_width, available_height)
            else:
                # Fallback size calculation based on grid size
                grid_sizes = {"1x1": 500, "2x2": 250, "3x3": 180, "4x4": 150}
                size = grid_sizes.get(self.grid_size.get(), 180)
            
            # Apply size limits
            size = max(size, 80)   # Minimum size
            size = min(size, 600)  # Maximum size
            
            # Resize image to fit available space
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((int(size), int(size)), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update label
            self.preview_image_labels[row][col].config(image=photo, text="")
            self.preview_image_labels[row][col].image = photo  # Keep reference
            
            # Add title as tooltip (simplified - just update the relief/background for now)
            if "No effects" in title:
                self.preview_image_labels[row][col].config(relief="solid", bg="lightgreen")
            else:
                self.preview_image_labels[row][col].config(relief="raised", bg="lightblue")
                
        except Exception as e:
            # Show error in cell
            if hasattr(self, 'preview_image_labels') and len(self.preview_image_labels) > row and len(self.preview_image_labels[row]) > col:
                self.preview_image_labels[row][col].config(
                    image="", text=f"Error\n{str(e)[:20]}...", bg="lightcoral"
                )
    
    def on_preview_resize(self, event):
        """Handle preview window resize - update image sizes."""
        # Only respond to window resize events, not individual widget resizes
        if event.widget != self.preview_window:
            return
            
        if hasattr(self, 'preview_images') and self.preview_images:
            # Small delay to avoid too frequent updates during dragging
            if hasattr(self, '_resize_timer'):
                self.preview_window.after_cancel(self._resize_timer)
            self._resize_timer = self.preview_window.after(200, self.update_preview_sizes)
    
    def update_preview_sizes(self):
        """Update all preview image sizes after window resize."""
        if not hasattr(self, 'preview_images') or not self.preview_images:
            return
        
        try:
            # Get current grid size
            grid_sizes = {"1x1": 1, "2x2": 4, "3x3": 9, "4x4": 16}
            max_images = grid_sizes[self.grid_size.get()]
            
            # Re-display all current images with new sizes
            for i, img in enumerate(self.preview_images[:max_images]):
                self.display_preview_image(img, i, "Resized")
                
        except Exception as e:
            print(f"Error during resize update: {e}")
    
    def update_model_info(self, event=None):
        """Update model size information."""
        size_info = {
            "n": "Nano (3.2M params) - Fastest",
            "s": "Small (11.2M params) - Fast", 
            "m": "Medium (25.9M params) - Balanced",
            "l": "Large (43.7M params) - Accurate",
            "x": "XLarge (68.2M params) - Most Accurate"
        }
        self.model_info.config(text=size_info.get(self.model_size.get(), "Unknown"))
        self.update_command_preview()
        
    def browse_dataset(self):
        """Browse for dataset directory."""
        directory = filedialog.askdirectory(title="Select Dataset Directory")
        if directory:
            self.dataset_path.set(directory)
            self.update_command_preview()
            
    def browse_model(self):
        """Browse for pretrained model file."""
        filename = filedialog.askopenfilename(
            title="Select Pretrained Model",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.pretrained_model.set(filename)
            self.update_command_preview()
            
    def update_command_preview(self):
        """Update the training command preview."""
        try:
            # Build model path
            if self.pretrained_model.get().strip():
                model_path = self.pretrained_model.get()
            else:
                model_path = f"yolov8{self.model_size.get()}.pt"
            
            # Build command
            cmd_parts = [
                "python", "src/train.py",
                f"--dataset \"{self.dataset_path.get()}\"",
                f"--model {self.model_size.get()}",
                f"--epochs {self.epochs.get()}",
                f"--batch {self.batch_size.get()}"
            ]
            
            # Add optional parameters that differ from defaults
            if self.image_size.get() != 640:
                cmd_parts.append(f"--imgsz {self.image_size.get()}")
            if self.device.get() != "auto":
                cmd_parts.append(f"--device {self.device.get()}")
                
            command = " \\\n  ".join(cmd_parts)
            
            # Add Python training parameters as comment
            command += "\n\n# Augmentation Ranges (applied randomly during training):\n"
            command += f"# - Rotation: {self.degrees_min.get():.1f}° to {self.degrees_max.get():.1f}°\n"
            command += f"# - Motion Blur: {self.motion_blur_prob.get():.0%} chance, length {self.motion_length_min.get()}-{self.motion_length_max.get()}\n"
            command += f"# - Gaussian Blur: {self.gaussian_blur_prob.get():.0%} chance, kernel {self.gaussian_kernel_min.get()}-{self.gaussian_kernel_max.get()}\n"
            command += f"# - Horizontal Flip: {self.fliplr_prob.get():.0%} chance\n"
            command += f"# - Output: {self.output_project.get()}/{self.output_name.get()}"
            
            self.command_text.delete("1.0", tk.END)
            self.command_text.insert("1.0", command)
            
        except Exception as e:
            self.command_text.delete("1.0", tk.END)
            self.command_text.insert("1.0", f"Error generating command: {str(e)}")
            
    def start_training(self):
        """Start the training process."""
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already running!")
            return
            
        # Validate inputs
        if not Path(self.dataset_path.get()).exists():
            messagebox.showerror("Error", "Dataset path does not exist!")
            return
            
        dataset_yaml = Path(self.dataset_path.get()) / "dataset.yaml"
        if not dataset_yaml.exists():
            messagebox.showerror("Error", "dataset.yaml not found in dataset directory!")
            return
            
        # Update status
        self.is_training = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Training in progress...", foreground="orange")
        
        # Clear output
        self.output_text.delete("1.0", tk.END)
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self.run_training, daemon=True)
        training_thread.start()
        
    def run_training(self):
        """Run the actual training process."""
        try:
            # Build command for subprocess
            cmd = [
                sys.executable, "src/train.py",
                "--dataset", self.dataset_path.get(),
                "--model", self.model_size.get(),
                "--epochs", str(self.epochs.get()),
                "--batch", str(self.batch_size.get())
            ]
            
            # Add device if not auto
            if self.device.get() != "auto":
                cmd.extend(["--device", self.device.get()])
            
            # Log start
            self.root.after(0, lambda: self.append_output("Starting training with command:\n"))
            self.root.after(0, lambda: self.append_output(" ".join(cmd) + "\n\n"))
            
            # Start subprocess
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output in real-time
            for line in iter(self.training_process.stdout.readline, ''):
                if not self.is_training:  # Check if stopped
                    break
                
                # Regular output
                self.root.after(0, lambda l=line: self.append_output(l))
            
            # Wait for process to complete
            self.training_process.wait()
            
            # Check result
            if self.training_process.returncode == 0:
                self.root.after(0, lambda: self.training_completed(True, "Training completed successfully!"))
            else:
                self.root.after(0, lambda: self.training_completed(False, f"Training failed with return code {self.training_process.returncode}"))
                
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.root.after(0, lambda: self.training_completed(False, error_msg))
            
    def training_completed(self, success, message):
        """Handle training completion."""
        self.is_training = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        if success:
            self.status_label.config(text="Training completed successfully!", foreground="green")
            self.append_output(f"\n{message}\n")
            self.append_output(f"Model saved to: {self.output_project.get()}/{self.output_name.get()}/weights/best.pt\n")
            messagebox.showinfo("Success", message)
        else:
            self.status_label.config(text="Training failed!", foreground="red")
            self.append_output(f"\nERROR: {message}\n")
            messagebox.showerror("Error", message)
            
    def stop_training(self):
        """Stop the training process."""
        if self.training_process and self.training_process.poll() is None:
            # Terminate the subprocess
            self.training_process.terminate()
            
            # Wait a bit, then force kill if needed
            try:
                self.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
                self.training_process.wait()
            
        self.is_training = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Training stopped by user", foreground="red")
        self.append_output("\n=== Training stopped by user ===\n")
        
    def append_output(self, text):
        """Append text to output area."""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    style.configure("Success.TButton", foreground="green")
    
    app = YOLOTrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()