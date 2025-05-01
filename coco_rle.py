import os
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import json

# --- Existing functions (remain mostly the same) ---

def load_mask(mask_path):
    """Loads a binary mask from a PNG file."""
    try:
        mask = np.array(Image.open(mask_path))
        # Ensure binary (handles potential grayscale or alpha channels)
        if mask.ndim == 3: # Handle cases where mask might be RGBA, RGB, or grayscale with alpha
             # Simple conversion: consider any non-zero pixel as part of the mask
             mask = np.any(mask > 0, axis=-1) if mask.shape[-1] > 1 else mask > 0
        else: # Grayscale
             mask = mask > 0

        return mask.astype(np.uint8)
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None # Return None if loading fails

def encode_rle(binary_mask):
    """Encodes a binary mask (H, W) into COCO RLE format."""
    # Ensure the mask is in Fortran order (column-major) for pycocotools
    binary_mask = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(binary_mask)
    # Decode bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def find_mask_paths(masks_root, image_name_prefix):
    """Finds all mask paths for a given image name prefix."""
    mask_paths = []
    # Construct the directory path expected for the image's masks
    image_mask_dir = os.path.join(masks_root, image_name_prefix)

    # Check if the directory exists
    if not os.path.isdir(image_mask_dir):
        # print(f"Warning: Mask directory not found for image '{image_name_prefix}': {image_mask_dir}")
        return [] # Return empty list if no mask directory

    for root, _, files in os.walk(image_mask_dir):
        for file in files:
            if file.lower().endswith('.png'): # Case-insensitive check
                mask_paths.append(os.path.join(root, file))

    # Optional: print found masks for verification
    # if not mask_paths:
    #     print(f"Warning: No .png masks found in '{image_mask_dir}'")

    return mask_paths

# --- New/Modified Processing Function ---

def process_dataset_instance_segmentation(images_dir, masks_root, output_json):
    """
    Processes images and individual masks to create a COCO-like instance segmentation JSON.

    Args:
        images_dir (str): Directory containing image files.
        masks_root (str): Root directory containing subdirectories for masks
                          (e.g., masks_root/image_name_prefix/mask1.png).
        output_json (str): Path to save the output JSON file.
    """
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
        # Optionally add 'info', 'licenses' if needed for full COCO spec
    }

    # Define categories - assuming only one category 'cell' with ID 1
    # If you have multiple categories, define them here
    categories = [{'id': 1, 'name': 'cell', 'supercategory': 'cell'}]
    coco_output['categories'] = categories
    cell_category_id = categories[0]['id'] # Get the ID for the 'cell' category

    image_id_counter = 1
    annotation_id_counter = 1

    # Get a sorted list of image files for consistent processing order
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Found {len(image_files)} image files in {images_dir}")

    for img_file in image_files:
        image_path = os.path.join(images_dir, img_file)
        image_name_prefix = os.path.splitext(img_file)[0] # Get filename without extension

        try:
            with Image.open(image_path) as img:
                 # Store PIL dimensions (W, H) and get NumPy shape (H, W)
                image_width, image_height = img.size
                image_shape = (image_height, image_width) # (H, W) for numpy operations
        except Exception as e:
            print(f"Error opening image {image_path}: {e}. Skipping.")
            continue

        # Find all individual mask paths for this image
        mask_paths = find_mask_paths(masks_root, image_name_prefix)

        # Create image entry
        image_info = {
            'id': image_id_counter,
            'file_name': img_file,
            'height': image_height,
            'width': image_width
        }
        coco_output['images'].append(image_info)

        print(f"Processing image '{img_file}' (ID: {image_id_counter}) with {len(mask_paths)} masks...")

        if not mask_paths:
            # print(f"No masks found for image '{img_file}'. Adding image entry without annotations.")
            pass # Image added, but no annotations will be added in the next loop

        # Process each individual mask
        for mask_path in mask_paths:
            binary_mask = load_mask(mask_path)

            if binary_mask is None:
                 print(f"Skipping invalid mask: {mask_path}")
                 continue # Skip this mask if loading failed

            # Ensure loaded mask matches expected image dimensions
            if binary_mask.shape != image_shape:
                print(f"Warning: Mask shape {binary_mask.shape} mismatch with image shape {image_shape} for {mask_path}. Skipping mask.")
                continue

            rle = encode_rle(binary_mask)

            # Calculate area and bbox from RLE
            area = float(mask_utils.area(rle))
            # pycocotools bbox format is [x, y, width, height]
            bbox = mask_utils.toBbox(rle).tolist() # Convert numpy array to list

            # Create annotation entry for this individual mask
            annotation = {
                'id': annotation_id_counter,          # Unique ID for this annotation
                'image_id': image_id_counter,         # Link to the image
                'category_id': cell_category_id,      # Category of the object (e.g., cell)
                'segmentation': rle,                  # The RLE mask data
                'area': area,                         # Area of the mask
                'bbox': bbox,                         # Bounding box [x,y,w,h]
                'iscrowd': 0                          # 0 for individual instances
            }

            coco_output['annotations'].append(annotation)
            annotation_id_counter += 1

        image_id_counter += 1

    # Save the final JSON structure
    print(f"Saving annotations to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=2)

    print("Processing complete.")

# Example usage:
if __name__ == "__main__":
    # # Create dummy directories and files for testing if they don't exist
    # if not os.path.exists('50_dataset/bf/image1.png'):
    #     print("Creating dummy data for testing...")
    #     os.makedirs('50_dataset/bf', exist_ok=True)
    #     os.makedirs('50_dataset/binary_masks/image1', exist_ok=True)
    #     os.makedirs('50_dataset/binary_masks/image2', exist_ok=True)

    #     # Create a dummy image
    #     dummy_img = Image.new('RGB', (100, 100), color='white')
    #     dummy_img.save('50_dataset/bf/image1.png')
    #     dummy_img.save('50_dataset/bf/image2.png')

    #     # Create dummy masks for image1
    #     mask1_1 = Image.new('L', (100, 100), color=0)
    #     mask1_2 = Image.new('L', (100, 100), color=0)
    #     mask1_3 = Image.new('L', (100, 100), color=0)

    #     # Draw simple shapes on masks (e.g., squares)
    #     # Mask 1_1: top-left square
    #     for x in range(10, 30):
    #         for y in range(10, 30):
    #             mask1_1.putpixel((x, y), 255)
    #     mask1_1.save('50_dataset/binary_masks/image1/mask_cell_001.png')

    #     # Mask 1_2: bottom-right square
    #     for x in range(70, 90):
    #         for y in range(70, 90):
    #             mask1_2.putpixel((x, y), 255)
    #     mask1_2.save('50_dataset/binary_masks/image1/mask_cell_002.png')

    #     # Mask 1_3: center square
    #     for x in range(40, 60):
    #          for y in range(40, 60):
    #               mask1_3.putpixel((x, y), 255)
    #     mask1_3.save('50_dataset/binary_masks/image1/mask_cell_003.png')


    #     # Create dummy masks for image2 (only one mask)
    #     mask2_1 = Image.new('L', (100, 100), color=0)
    #      # Mask 2_1: large rectangle
    #     for x in range(20, 80):
    #         for y in range(30, 70):
    #             mask2_1.putpixel((x, y), 255)
    #     mask2_1.save('50_dataset/binary_masks/image2/mask_cell_001.png')

    #     print("Dummy data created.")

    # --- Actual Usage ---
    process_dataset_instance_segmentation(
        images_dir='50_dataset/bf',
        masks_root='50_dataset/binary_masks',
        output_json='individual_rle_annotations.json'
    )

    print("\nCheck 'individual_rle_annotations.json' for the output.")