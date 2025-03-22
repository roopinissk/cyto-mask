import os
import glob
import cv2
import numpy as np

# Define directories
binary_masks_dir = '/home/suriya/cyto-mask/dataset/binary_masks'
merged_binary_dir = "merged_binary_cells"
edge_detected_dir = "edge_detected_cells"

# Create output directories if they don't exist
os.makedirs(merged_binary_dir, exist_ok=True)
os.makedirs(edge_detected_dir, exist_ok=True)

# Process each subfolder in the binary masks directory
dirs = os.listdir(binary_masks_dir)
for i in dirs:
    folder_path = os.path.join(binary_masks_dir, i)
    
    # Get all PNG mask file paths in the subfolder
    mask_paths = glob.glob(os.path.join(folder_path, "*.png"))
    if not mask_paths:
        continue

    # Merge binary masks by taking the maximum pixel value at each location
    merged_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    for mask_path in mask_paths[1:]:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to read mask: {mask_path}")
            continue
        merged_mask = np.maximum(merged_mask, mask)
    
    # Save the merged binary mask in its dedicated directory
    cv2.imwrite(os.path.join(merged_binary_dir, f"merged_{i}.png"), merged_mask)

    # Create a subdirectory for edge detection images for this mask folder
    edge_subdir = os.path.join(edge_detected_dir, i)
    os.makedirs(edge_subdir, exist_ok=True)

    # Process each mask file to compute and save edge detection images
    for mask_file in os.listdir(folder_path):
        if mask_file.endswith(".png"):
            mask_path = os.path.join(folder_path, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to read mask: {mask_path}")
                continue
            edges = cv2.Canny(mask, 100, 200)  # Adjust thresholds as needed
            edges_file_name = f"edges_{mask_file}"
            cv2.imwrite(os.path.join(edge_subdir, edges_file_name), edges)
            print(f"Processed {mask_file} and saved edges as {edges_file_name}")

    # Merge the individual edge images in the subfolder
    edge_paths = glob.glob(os.path.join(edge_subdir, "edges_*.png"))
    if not edge_paths:
        continue
    merged_edges = cv2.imread(edge_paths[0], cv2.IMREAD_GRAYSCALE)
    for edge_path in edge_paths[1:]:
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if edge is None:
            print(f"Failed to read edge: {edge_path}")
            continue
        merged_edges = np.maximum(merged_edges, edge)
    
    # Save the merged edge image in the corresponding subdirectory
    cv2.imwrite(os.path.join(edge_subdir, f"merged_edges_{i}.png"), merged_edges)
