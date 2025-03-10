import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Input and output directories
INPUT_FOLDER = "raw_images"
OUTPUT_FOLDER = "processed_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to create a transparent PNG
def make_transparent(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, alpha = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # Extract dark pixels
    
    # Create a 4-channel (RGBA) image
    b, g, r = cv2.split(img)  # Split channels
    rgba = cv2.merge((b, g, r, alpha))  # Merge with alpha
    return rgba

# List of preprocessing methods
preprocessing_methods = {
    "Grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    "Binarized": lambda img: cv2.adaptiveThreshold(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    ),
    "Noise_Reduced": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "Contrast_Enhanced": lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ),
    "Normalized": lambda img: (img / 255.0).astype(np.float32)
}

# Augmentation methods (Now Includes Mirroring & Flipping)
augmentations = {
    "Rotation": A.Rotate(limit=15, p=1.0),
    "Rotation_90": A.Rotate(limit=[90, 90], p=1.0),
    "Rotation_270": A.Rotate(limit=[270, 270], p=1.0),
    "Affine": A.Affine(scale=(0.8, 1.2), translate_percent=(0.05, 0.1), shear=(-10, 10), p=1.0),
    "Elastic_Distortion": A.ElasticTransform(alpha=1, sigma=50, p=1.0),
    "Brightness_Contrast": A.RandomBrightnessContrast(p=1.0),
    "Gaussian_Noise": A.GaussNoise(p=1.0),
    "Grid_Distortion": A.GridDistortion(p=1.0),
    "Perspective_Transform": A.Perspective(scale=(0.05, 0.1), p=1.0),
    "Cutout": A.CoarseDropout(max_holes=2, hole_size=(20, 20), p=1.0),
    "Color_Jitter": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
    "Inversion": A.InvertImg(p=1.0),
    "Horizontal_Scaling": A.Resize(height=224, width=256, p=1.0),
    "Vertical_Scaling": A.Resize(height=256, width=224, p=1.0),
    "Mirror_Horizontal": A.HorizontalFlip(p=1.0), 
    "Flip_Vertical": A.VerticalFlip(p=1.0)
}

# Process all images
for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    
    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    if img is None:
        print(f"Skipping {filename}, invalid image file.")
        continue

    base_filename = os.path.splitext(filename)[0] + ".png"

    # 🔹 Step 1: Convert to PNG and save in raw format
    raw_png_path = os.path.join(OUTPUT_FOLDER, "Raw_PNG")
    os.makedirs(raw_png_path, exist_ok=True)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(os.path.join(raw_png_path, base_filename))

    # 🔹 Step 2: Create transparent PNG and save
    transparent_img = make_transparent(img)
    transparent_folder = os.path.join(OUTPUT_FOLDER, "Transparent_PNG")
    os.makedirs(transparent_folder, exist_ok=True)
    cv2.imwrite(os.path.join(transparent_folder, base_filename), transparent_img)

    # 🔹 Step 3: Apply preprocessing
    for method, func in preprocessing_methods.items():
        processed_img = func(img)
        folder_path = os.path.join(OUTPUT_FOLDER, method)
        os.makedirs(folder_path, exist_ok=True)
        
        if isinstance(processed_img, np.ndarray):
            if len(processed_img.shape) == 2:  # Grayscale
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(os.path.join(folder_path, base_filename), processed_img * 255 if "Normalized" in method else processed_img)

    # 🔹 Step 4: Apply augmentations (Now Includes Mirroring & Flipping)
    for aug_name, transform in augmentations.items():
        augmented = transform(image=img)['image']
        aug_folder_path = os.path.join(OUTPUT_FOLDER, aug_name)
        os.makedirs(aug_folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(aug_folder_path, base_filename), augmented)

print("✅ Preprocessing & Augmentation Completed! Mirroring & Flipping Added.") 
