"""
Interactive Blur Parameter Tuning GUI
Real-time blur effect adjustment for training data augmentation
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time


class BlurTunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Blur Tuner for Training Data")
        self.root.geometry("1200x800")
        
        # Variables
        self.original_image = None
        self.current_blurred = None
        self.image_path = tk.StringVar()
        
        # Blur parameters
        self.gaussian_kernel = tk.IntVar(value=15)
        self.gaussian_sigma = tk.DoubleVar(value=5.0)
        self.motion_length = tk.IntVar(value=15)
        self.motion_angle = tk.DoubleVar(value=45.0)
        self.radial_strength = tk.IntVar(value=10)
        self.noise_strength = tk.IntVar(value=25)
        
        # Auto-update flag
        self.auto_update = tk.BooleanVar(value=False)
        self.last_update_time = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Entry(file_frame, textvariable=self.image_path, width=60).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(file_frame, text="Browse", command=self.browse_image).grid(row=0, column=1)
        ttk.Button(file_frame, text="Load", command=self.load_image).grid(row=0, column=2, padx=(5, 0))
        
        # Parameters panel
        params_frame = ttk.LabelFrame(main_frame, text="Blur Parameters", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Blur type selection (multiple checkboxes)
        ttk.Label(params_frame, text="Blur Types (can combine):").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Individual blur type variables
        self.enable_gaussian = tk.BooleanVar(value=True)
        self.enable_motion = tk.BooleanVar(value=False)
        self.enable_radial = tk.BooleanVar(value=False)
        self.enable_noise = tk.BooleanVar(value=False)
        
        blur_checkboxes = [
            ("Gaussian Blur", self.enable_gaussian),
            ("Motion Blur", self.enable_motion), 
            ("Radial Blur", self.enable_radial),
            ("Noise + Blur", self.enable_noise)
        ]
        
        for i, (text, var) in enumerate(blur_checkboxes):
            ttk.Checkbutton(params_frame, text=text, variable=var, 
                           command=self.on_param_change).grid(row=i+1, column=0, sticky=tk.W)
        
        # Gaussian Blur Parameters
        gaussian_frame = ttk.LabelFrame(params_frame, text="Gaussian Blur", padding="5")
        gaussian_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(10, 5))
        
        ttk.Label(gaussian_frame, text="Kernel Size:").grid(row=0, column=0, sticky=tk.W)
        kernel_scale = ttk.Scale(gaussian_frame, from_=3, to=51, orient=tk.HORIZONTAL, 
                                variable=self.gaussian_kernel, command=self.on_param_change)
        kernel_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Label(gaussian_frame, textvariable=self.gaussian_kernel).grid(row=0, column=2)
        
        ttk.Label(gaussian_frame, text="Sigma:").grid(row=1, column=0, sticky=tk.W)
        sigma_scale = ttk.Scale(gaussian_frame, from_=0.1, to=20.0, orient=tk.HORIZONTAL,
                               variable=self.gaussian_sigma, command=self.on_param_change)
        sigma_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Label(gaussian_frame, text=f"{self.gaussian_sigma.get():.1f}").grid(row=1, column=2)
        
        # Motion Blur Parameters
        motion_frame = ttk.LabelFrame(params_frame, text="Motion Blur", padding="5")
        motion_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(motion_frame, text="Length:").grid(row=0, column=0, sticky=tk.W)
        length_scale = ttk.Scale(motion_frame, from_=3, to=50, orient=tk.HORIZONTAL,
                                variable=self.motion_length, command=self.on_param_change)
        length_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Label(motion_frame, textvariable=self.motion_length).grid(row=0, column=2)
        
        ttk.Label(motion_frame, text="Angle:").grid(row=1, column=0, sticky=tk.W)
        angle_scale = ttk.Scale(motion_frame, from_=0, to=180, orient=tk.HORIZONTAL,
                               variable=self.motion_angle, command=self.on_param_change)
        angle_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Label(motion_frame, text=f"{self.motion_angle.get():.0f}°").grid(row=1, column=2)
        
        # Radial Blur Parameters
        radial_frame = ttk.LabelFrame(params_frame, text="Radial Blur", padding="5")
        radial_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(radial_frame, text="Strength:").grid(row=0, column=0, sticky=tk.W)
        radial_scale = ttk.Scale(radial_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                variable=self.radial_strength, command=self.on_param_change)
        radial_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Label(radial_frame, textvariable=self.radial_strength).grid(row=0, column=2)
        
        # Noise Blur Parameters
        noise_frame = ttk.LabelFrame(params_frame, text="Noise + Blur", padding="5")
        noise_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(noise_frame, text="Noise Strength:").grid(row=0, column=0, sticky=tk.W)
        noise_scale = ttk.Scale(noise_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                               variable=self.noise_strength, command=self.on_param_change)
        noise_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        ttk.Label(noise_frame, textvariable=self.noise_strength).grid(row=0, column=2)
        
        # Control buttons
        control_frame = ttk.Frame(params_frame)
        control_frame.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        
        ttk.Checkbutton(control_frame, text="Auto Update", 
                       variable=self.auto_update).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="Recompute", 
                  command=self.recompute_blur).grid(row=0, column=1, padx=(20, 0))
        
        ttk.Button(control_frame, text="Reset", 
                  command=self.reset_parameters).grid(row=0, column=2, padx=(10, 0))
        
        # Training suggestions
        suggestions_frame = ttk.LabelFrame(params_frame, text="Training Suggestions", padding="5")
        suggestions_frame.grid(row=11, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        
        suggestions_text = tk.Text(suggestions_frame, height=8, width=40, wrap=tk.WORD)
        suggestions_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        suggestions = """Webcam Motion Blur:
• Gaussian: kernel=7-21, sigma=2-8
• Motion: length=5-25, any angle
• Use multiple random values

Low Light Conditions:
• Noise + Blur: noise=10-50
• Gaussian: small kernel, low sigma

Good Starting Values:
• Gaussian: kernel=11, sigma=3
• Motion: length=15, angle=30
• Noise: strength=20"""
        
        suggestions_text.insert("1.0", suggestions)
        suggestions_text.config(state="disabled")
        
        # Image display
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg="white", width=600, height=600)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Configure canvas grid
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Configure main grid
        for frame in [gaussian_frame, motion_frame, radial_frame, noise_frame, control_frame]:
            frame.columnconfigure(1, weight=1)
    
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.image_path.set(filename)
            self.load_image()
    
    def load_image(self):
        path = self.image_path.get()
        if not path:
            messagebox.showerror("Error", "Please select an image file!")
            return
        
        try:
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                raise ValueError("Could not load image")
            
            # Display original image
            self.display_image(self.original_image)
            messagebox.showinfo("Success", f"Image loaded successfully!\nSize: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def on_param_change(self, *args):
        current_time = time.time()
        self.last_update_time = current_time
        
        if self.auto_update.get():
            # Delay auto-update to avoid too frequent computations
            self.root.after(500, lambda: self.delayed_update(current_time))
    
    def delayed_update(self, trigger_time):
        if time.time() - self.last_update_time < 0.4:  # Still changing
            return
        if trigger_time == self.last_update_time:  # This is the latest change
            self.recompute_blur()
    
    def recompute_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        # Start with original image
        result = self.original_image.copy()
        applied_effects = []
        
        try:
            # Apply Gaussian blur if enabled
            if self.enable_gaussian.get():
                kernel = self.gaussian_kernel.get()
                if kernel % 2 == 0:
                    kernel += 1
                sigma = self.gaussian_sigma.get()
                result = cv2.GaussianBlur(result, (kernel, kernel), sigma)
                applied_effects.append(f"Gaussian(k={kernel},σ={sigma:.1f})")
            
            # Apply Motion blur if enabled
            if self.enable_motion.get():
                length = self.motion_length.get()
                angle = self.motion_angle.get()
                result = self.apply_motion_blur(result, length, angle)
                applied_effects.append(f"Motion(l={length},a={angle:.0f}°)")
            
            # Apply Radial blur if enabled
            if self.enable_radial.get():
                strength = self.radial_strength.get()
                result = self.apply_radial_blur(result, strength)
                applied_effects.append(f"Radial(s={strength})")
            
            # Apply Noise blur if enabled
            if self.enable_noise.get():
                noise_strength = self.noise_strength.get()
                result = self.add_noise_blur(result, noise_strength)
                applied_effects.append(f"Noise(s={noise_strength})")
            
            self.current_blurred = result
            
            # Create title with applied effects
            if applied_effects:
                effects_title = " + ".join(applied_effects)
            else:
                effects_title = "No effects applied"
                result = self.original_image  # Show original if no effects
            
            self.display_comparison(self.original_image, result, effects_title)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply blur: {str(e)}")
    
    def apply_motion_blur(self, image, length, angle):
        """Apply motion blur effect."""
        kernel = np.zeros((length, length))
        center = length // 2
        angle_rad = np.radians(angle)
        
        for i in range(length):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        kernel = kernel / np.sum(kernel)
        return cv2.filter2D(image, -1, kernel)
    
    def apply_radial_blur(self, image, strength):
        """Apply radial blur effect."""
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
                kernel_size = blur_level * 2 + 1
                blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_level)
                result[mask] = blurred[mask]
        
        return result
    
    def add_noise_blur(self, image, noise_strength):
        """Add noise and blur effect."""
        noise = np.random.normal(0, noise_strength, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(noisy, (3, 3), 1.0)
    
    def display_image(self, image):
        """Display single image on canvas."""
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Resize if too large
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
            image_width, image_height = pil_image.size
            scale = min(canvas_width / image_width, canvas_height / image_height, 1.0)
            
            if scale < 1.0:
                new_width = int(image_width * scale)
                new_height = int(image_height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def display_comparison(self, original, blurred, effects_title="Blurred"):
        """Display original and blurred images side by side."""
        # Resize images to fit canvas
        height = min(original.shape[0], 400)
        width = int(original.shape[1] * height / original.shape[0])
        
        original_resized = cv2.resize(original, (width, height))
        blurred_resized = cv2.resize(blurred, (width, height))
        
        # Create side-by-side comparison
        comparison = np.hstack([original_resized, blurred_resized])
        
        # Convert to RGB
        rgb_comparison = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rgb_comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        
        # Truncate effects title if too long
        if len(effects_title) > 30:
            effects_title = effects_title[:27] + "..."
        
        cv2.putText(rgb_comparison, effects_title, 
                   (width + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Convert to PIL and display
        pil_image = Image.fromarray(rgb_comparison)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.gaussian_kernel.set(15)
        self.gaussian_sigma.set(5.0)
        self.motion_length.set(15)
        self.motion_angle.set(45.0)
        self.radial_strength.set(10)
        self.noise_strength.set(25)
        self.blur_type.set("gaussian")
        
        if self.original_image is not None:
            self.recompute_blur()


def main():
    root = tk.Tk()
    app = BlurTunerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()