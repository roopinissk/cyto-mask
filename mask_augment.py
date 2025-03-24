import augly.image as imaugs
import os
import random
import glob
from tqdm import tqdm
from PIL import Image
import json
import shutil  # Added missing import

# Set random seed for reproducibility
random.seed(0)

# Load rotation degrees from file
with open('rotation_degrees.json', 'r') as f:
    rotation_degrees = json.load(f)

adjusted_rotation_degrees = {f"merged_edges_{key}": value for key, value in rotation_degrees.items()}

print("Keys in adjusted_rotation_degrees:", adjusted_rotation_degrees.keys())

# augment_path = "dataset/augment"
augment_ground_truth_path = "/home/suriya/cyto-mask/merged_edge_detected_cells"
output_dir = "dataset/augment_ground_truth"
os.makedirs(output_dir, exist_ok=True)

def rotate(augment_ground_truth_path, output_dir, degrees):
    base = os.path.basename(augment_ground_truth_path)
    base_name, _ = os.path.splitext(base)
    new_filename = f"{base_name}_rotated{degrees}_.png"
    new_img_path = os.path.join(output_dir, new_filename)
    
    
    imaugs.rotate(image=augment_ground_truth_path, output_path=new_img_path, degrees=degrees)
    return new_img_path

def augment_gr_images(augment_ground_truth_path, output_dir="dataset/augment_ground_truth"):
    os.makedirs(output_dir, exist_ok=True)
    gt_images = sorted(glob.glob(os.path.join(augment_ground_truth_path, "*.png")))
    print(f"Found {len(gt_images)} ground truth images in {augment_ground_truth_path}")
    
    for img_path in tqdm(gt_images, desc="Processing images"):
        base = os.path.basename(img_path)
        degrees = adjusted_rotation_degrees.get(base)
        if degrees is not None:
            rotate(img_path, output_dir, degrees)
        else:
            print(f"No rotation degree found for {base}")

# Call the function
augment_gr_images("/home/suriya/cyto-mask/merged_edge_detected_cells", output_dir="dataset/augment_ground_truth")