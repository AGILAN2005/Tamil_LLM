import os
import shutil

# List of directories containing images
image_dirs = [
    'Affine', 'Binarized', 'Brightness_Contrast', 'Color_Jitter', 'Contrast_Enhanced',
    'Cutout', 'Elastic_Distortion', 'Gaussian_Noise', 'Grayscale', 'Grid_Distortion',
    'Inversion', 'Noise_Reduced', 'Normalized', 'Perspective_Transform', 'Raw_PNG',
    'Rotation', 'Rotation_90', 'Rotation_270', 'Transparent_PNG', 'Horizontal_Scaling', 
    'Vertical_Scaling', 'Mirror_Horizontal', 'Flip_Vertical'
]

# Path where your directories are located (update this with your actual path)
base_path = "processed_images"

# Create a new folder to store the final results (if not exists)
output_base_dir = os.path.join(base_path, 'sorted_images')
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Loop through the directories
for image_dir in image_dirs:
    image_dir_path = os.path.join(base_path, image_dir)
    
    # Loop through all files in the current image directory
    for filename in os.listdir(image_dir_path):
        # Get the full path of the image
        file_path = os.path.join(image_dir_path, filename)

        # Skip directories, process only files
        if os.path.isdir(file_path):
            continue
        
        # Get the base name of the file without the extension (this will be used for the folder name)
        image_name, ext = os.path.splitext(filename)

        # Create a folder for this image name in the output folder
        target_folder = os.path.join(output_base_dir, image_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Determine the serial number for the file
        serial_number = len([f for f in os.listdir(target_folder) if f.endswith(ext)]) + 1
        new_filename = f"{serial_number}{ext}"
        
        # Copy the image to the target folder with a new name
        shutil.copy(file_path, os.path.join(target_folder, new_filename))

print("Images have been sorted and copied into corresponding folders with serialized names.")
