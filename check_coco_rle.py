import json
import sys
import os
import numpy as np
from PIL import Image # We need Pillow to save the mask

# Optional: Try importing pycocotools for RLE decoding and strict check
PYCOCOTOOLS_AVAILABLE = False
try:
    from pycocotools import mask as maskUtils
    PYCOCOTOOLS_AVAILABLE = True
    # print("Info: pycocotools found, RLE decoding and strict check available.")
except ImportError:
    # print("Warning: pycocotools not found. RLE decoding and strict check will be skipped.")
    pass

# --- Modified Validation Function for Standard COCO Structure ---

def check_coco_annotations_rle_format(json_path):
    """
    Checks if a JSON file following the standard COCO format,
    specifically the 'annotations' list, contains valid RLE segmentation data.

    Checks for:
    1. Top level is a dictionary.
    2. Contains an 'annotations' key, and its value is a list.
    3. Each item in the 'annotations' list is a dictionary.
    4. If an annotation dictionary contains a 'segmentation' key and its value is a dictionary (RLE):
       - The 'segmentation' dictionary contains 'size' and 'counts' keys.
       - 'size' is a list of exactly two non-negative integers [height, width].
       - 'counts' is a string.
       - (Optional, if pycocotools is available) The 'counts' string can be decoded as valid RLE.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        tuple: A tuple containing:
               - bool: True if all checked RLE segmentations are compliant within 'annotations', False otherwise.
                       Returns True if the file is empty, invalid structure, or contains no 'segmentation' dictionaries.
               - list: A list of error messages detailing non-compliant items.
               - dict or None: The loaded COCO dictionary if successful, otherwise None.
    """
    errors = []
    data = None

    # 1. Load the JSON file
    if not os.path.exists(json_path):
        return False, [f"Error: File not found at {json_path}"], None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Error: Invalid JSON format in {json_path}: {e}"], None
    except Exception as e:
        return False, [f"Error: Could not open or read file {json_path}: {e}"], None

    # 2. Ensure the top level is a dictionary (standard COCO)
    if not isinstance(data, dict):
        return False, [f"Error: Top level of the JSON file is not a dictionary (expected COCO format). Found type: {type(data)}."], None

    # 3. Check for the 'annotations' list
    if 'annotations' not in data:
        # If no annotations key, maybe it's just an empty dataset, technically compliant
        # or just image/category info. We check the structure we care about.
        print("Warning: 'annotations' key not found in the top-level dictionary.")
        # Assuming this is okay if no annotations are expected
        return True, [], data # Return data for potential visualization

    annotations_list = data['annotations']

    if not isinstance(annotations_list, list):
        return False, [f"Error: Value for 'annotations' key is not a list. Found type: {type(annotations_list)}."], data

    if not annotations_list:
        # An empty annotations list is technically compliant
        return True, [], data

    # 4. Iterate through annotations and check 'segmentation' format
    is_compliant = True
    for i, anno_item in enumerate(annotations_list):
        # Ensure annotation item is a dictionary
        if not isinstance(anno_item, dict):
            errors.append(f"Annotations item {i} in list: Not a dictionary. Found type: {type(anno_item)}.")
            is_compliant = False
            continue # Can't check further on this item

        # Check for 'segmentation' key (COCO RLE uses 'segmentation')
        if 'segmentation' in anno_item:
            seg_data = anno_item['segmentation']

            # The value of 'segmentation' should be the RLE dictionary for RLE format
            if isinstance(seg_data, dict):
                # This should be RLE, check its format

                # Check for required RLE keys: 'size' and 'counts'
                if 'size' not in seg_data or 'counts' not in seg_data:
                    errors.append(f"Annotations item {i}: 'segmentation' dictionary missing required keys ('size' and 'counts'). Found keys: {list(seg_data.keys())}")
                    is_compliant = False
                    # Continue to next item, as structure is fundamentally wrong
                    continue

                # Check format of 'size'
                size = seg_data['size']
                if not (isinstance(size, list) and len(size) == 2 and
                        isinstance(size[0], int) and isinstance(size[1], int) and
                        size[0] >= 0 and size[1] >= 0): # Added non-negative check
                    errors.append(f"Annotations item {i}: 'segmentation.size' should be a list of two non-negative integers [height, width]. Found: {size} (type: {type(size)}, elements: {[type(e) for e in size] if isinstance(size, list) else 'N/A'}).")
                    is_compliant = False
                    # Cannot perform strict decode check if size is invalid

                # Check format of 'counts'
                counts = seg_data['counts']
                if not isinstance(counts, str):
                    errors.append(f"Annotations item {i}: 'segmentation.counts' should be a string. Found: {counts} (type: {type(counts)}).")
                    is_compliant = False
                else:
                    # Optional: Strict RLE decoding check using pycocotools
                    if PYCOCOTOOLS_AVAILABLE:
                        # Only attempt decode if basic size format seems correct
                        if isinstance(size, list) and len(size) == 2 and isinstance(size[0], int) and isinstance(size[1], int):
                             try:
                                # pycocotools.mask.decode expects the standard RLE object itself
                                rle_obj_for_decode = {'size': size, 'counts': counts}
                                # Try decoding - this will raise an error for invalid RLE strings
                                maskUtils.decode(rle_obj_for_decode) # We discard the output mask here
                             except Exception as e:
                                errors.append(f"Annotations item {i}: 'segmentation.counts' string is malformed or cannot be decoded by pycocotools. Error: {e}")
                                is_compliant = False
                        # else: size was already flagged as an error, skip decode attempt

            # Note: COCO 'segmentation' can also be a list of polygons.
            # If it's not a dict (RLE) and not a list (polygon), it's invalid for standard COCO segmentation.
            # If it *is* a list, this checker is currently focused only on RLE,
            # so we'll just note that it's not RLE if we wanted strict RLE only.
            # For now, we only report errors if 'segmentation' is present but *not* a dictionary AND not a list.
            elif not isinstance(seg_data, list) and seg_data is not None: # 'segmentation' exists, but its value is not a dictionary (RLE) or a list (Polygon) or None
                 errors.append(f"Annotations item {i}: 'segmentation' field is not a dictionary (RLE) or a list (Polygon). Found type: {type(seg_data)}.")
                 is_compliant = False

            # If 'segmentation' key is not present, it means this item doesn't have a mask annotation.
            # This is valid for COCO (e.g., a bbox annotation without segmentation). We do nothing in this case.

    return is_compliant, errors, data # Return the loaded dictionary (data)


# --- Modified Save Function for Standard COCO Structure ---

def save_first_rle_mask_from_coco(coco_data, output_dir="."):
    """
    Finds the first item in the 'annotations' list with RLE segmentation,
    decodes it, and saves the resulting binary mask as a grayscale PNG image.

    Args:
        coco_data (dict): The dictionary loaded from the COCO JSON file.
        output_dir (str): Directory to save the output mask image.
    """
    if not PYCOCOTOOLS_AVAILABLE:
        print("Cannot save mask: pycocotools library is required for RLE decoding.")
        return

    if not isinstance(coco_data, dict) or 'annotations' not in coco_data or not isinstance(coco_data['annotations'], list):
         print("Cannot attempt mask visualization: Input data is not a valid COCO structure with an annotations list.")
         return

    annotations_list = coco_data['annotations']

    first_rle_anno = None
    anno_index = -1
    image_filename = "unknown_image"
    for i, anno_item in enumerate(annotations_list):
        if isinstance(anno_item, dict) and 'segmentation' in anno_item and isinstance(anno_item['segmentation'], dict):
            # Found the first potential RLE annotation
            first_rle_anno = anno_item
            anno_index = i
            # Try to find the associated image filename for better naming
            image_id = anno_item.get('image_id')
            if image_id is not None and 'images' in coco_data and isinstance(coco_data['images'], list):
                 for img_info in coco_data['images']:
                      if isinstance(img_info, dict) and img_info.get('id') == image_id:
                           image_filename = img_info.get('file_name', f'image_id_{image_id}_no_name')
                           break # Found image info
            break # Stop after finding the first one

    if not first_rle_anno:
        print("No RLE segmentation found in the 'annotations' list to visualize.")
        return

    # We found an RLE annotation, try to decode and save it
    rle_obj = first_rle_anno['segmentation']
    size = rle_obj.get('size') # Use .get for safety, though compliance check passed
    counts = rle_obj.get('counts') # Use .get for safety

    # Perform basic size/counts check before decoding again (redundant if compliance check passed, but safe)
    if not (isinstance(size, list) and len(size) == 2 and
            isinstance(size[0], int) and isinstance(size[1], int) and
            size[0] >= 0 and size[1] >= 0 and isinstance(counts, str)):
        print(f"Cannot decode/save mask for annotation index {anno_index} (Image: '{image_filename}'): Basic RLE structure (size/counts type) is invalid.")
        return

    try:
        # Decode the RLE mask
        mask_array = maskUtils.decode({'size': size, 'counts': counts})

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the mask as an image using Pillow
        # Convert the boolean/uint8 mask to uint8 with values 0 and 255 for visibility
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L') # 'L' is 8-bit grayscale

        # Construct filename using image identifier and annotation index
        base_filename = os.path.splitext(image_filename)[0]
        output_filename = f"mask_first_rle_{base_filename}_anno_{anno_index}.png"
        output_path = os.path.join(output_dir, output_filename)

        mask_img.save(output_path)
        print(f"\nSuccessfully decoded and saved the first RLE mask for annotation index {anno_index} (Image: '{image_filename}') to: {output_path}")

    except Exception as e:
        print(f"\nError decoding or saving the first RLE mask for annotation index {anno_index} (Image: '{image_filename}'): {e}")


# --- Example Usage ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    print(f"Checking COCO RLE format in file: {json_file_path}")
    if PYCOCOTOOLS_AVAILABLE:
        print("pycocotools found. Strict RLE decoding check and mask visualization available.")
    else:
        print("pycocotools not found. Performing basic RLE structure/type checks only. Mask visualization is not available.")

    # Use the modified function that understands COCO structure
    compliant, issues, coco_data = check_coco_annotations_rle_format(json_file_path)

    if compliant:
        print("\nCompliance Result: The 'segmentation' sections within the 'annotations' list appear to follow the standard COCO RLE format.")
    else:
        print("\nCompliance Result: Issues found in the 'segmentation' sections within 'annotations':")
        for issue in issues:
            print(f"- {issue}")

    # Attempt to save the first RLE mask found using the COCO data structure
    if coco_data is not None:
         # Use the loaded COCO dictionary returned by the compliance check
         save_first_rle_mask_from_coco(coco_data, output_dir=".") # Saves in the current directory
    # else: Error already reported during load/initial check