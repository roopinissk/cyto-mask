import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection # Although we don't directly use CocoDetection, pycocotools.coco is needed
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm
import random

# Assuming you have Hugging Face transformers installed
from transformers import SamModel, SamProcessor # Or potentially a Sam2Model/Sam2Processor

# Assuming pycocotools is installed for RLE decoding
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO # Useful for loading annotations

# --- Configuration and Hyperparameters ---
CONFIG = {
    "images_dir": "50_dataset/bf",  # Directory containing your images
    "annotations_file_path": "individual_rle_annotations.json", # Your COCO RLE JSON file
    # Use a SAM-1 model name that works with transformers from_pretrained for now
    # until facebook/sam2-hiera-large is compatible or you implement manual loading
    "model_checkpoint": "facebook/sam-vit-base", # Or 'facebook/sam-vit-large', 'facebook/sam-vit-huge'
    "output_dir": "sam_finetuned_checkpoint", # Directory to save checkpoints

    "num_epochs": 10,
    "batch_size": 8, # Adjust based on GPU memory
    "learning_rate": 1e-5, # Learning rate for fine-tuning (usually small)
    "mask_decoder_learning_rate": 1e-4, # Higher LR for the mask decoder
    "weight_decay": 0.0001,
    "num_workers": 4, # Number of data loading workers

    "image_size": 1024, # SAM's default input size
    "prompt_type": "bbox", # How to generate prompts: 'bbox' or 'random_point' or 'multiple_points'
    "num_points_prompt": 1, # Number of random points if prompt_type is 'random_point' or 'multiple_points'

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# --- Custom Dataset for COCO and SAM Processing ---

class CocoSamDataset(Dataset):
    def __init__(self, images_dir, annotations_file_path, processor, prompt_type="bbox", num_points_prompt=1):
        self.images_dir = images_dir
        self.processor = processor
        self.prompt_type = prompt_type
        self.num_points_prompt = num_points_prompt

        # Use pycocotools to load annotations
        self.coco = COCO(annotations_file_path)
        # Get all image ids that have annotations
        self.img_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())

        # Create a list of (image_id, annotation_id) pairs
        # This ensures we have one item in the dataset per *annotation*
        self.annotation_pairs = []
        for img_id in self.img_ids:
            anno_ids = self.coco.getAnnIds(imgIds=img_id)
            # Filter annotations to only include those with segmentation (if you want)
            anno_ids_with_segmentation = [
                anno_id for anno_id in anno_ids
                if 'segmentation' in self.coco.loadAnns(anno_id)[0]
                and self.coco.loadAnns(anno_id)[0]['segmentation'] is not None
                and (isinstance(self.coco.loadAnns(anno_id)[0]['segmentation'], list) or isinstance(self.coco.loadAnns(anno_id)[0]['segmentation'], dict)) # Ensure it's list (polygon) or dict (RLE)
            ]
            for anno_id in anno_ids_with_segmentation:
                 self.annotation_pairs.append((img_id, anno_id))

        print(f"Loaded {len(self.annotation_pairs)} annotations across {len(self.img_ids)} images.")
        if len(self.annotation_pairs) == 0:
             print("Warning: No annotations with segmentation found. Dataset is empty.")


    def __len__(self):
        return len(self.annotation_pairs)

    def __getitem__(self, idx):
        img_id, anno_id = self.annotation_pairs[idx]

        try:
            # Load image
            img_info = self.coco.loadImgs(img_id)[0]
            image_path = os.path.join(self.images_dir, img_info['file_name'])
            # Check if image file exists
            if not os.path.exists(image_path):
                 print(f"Warning: Image file not found: {image_path}. Skipping annotation {anno_id}.")
                 return None # Return None if image not found
            image = Image.open(image_path).convert("RGB") # Ensure RGB

            # Load annotation
            annotation = self.coco.loadAnns(anno_id)[0]

            # Decode segmentation mask
            # annToMask handles both RLE (dict) and polygon (list)
            # It returns a numpy array (H, W) with values 0 or 1
            mask = self.coco.annToMask(annotation)

            # Ensure mask is binary uint8 (0 or 1)
            mask = (mask > 0).astype(np.uint8)

            # Generate Prompt (using bbox or points derived from the mask)
            input_boxes = None
            input_points = None

            if self.prompt_type == "bbox":
                # Bbox format [x, y, w, h], convert to [x1, y1, x2, y2]
                # Ensure bbox exists, although COCO format implies it should with segmentation
                if 'bbox' not in annotation or annotation['bbox'] is None:
                     print(f"Warning: Annotation {anno_id} is missing 'bbox'. Skipping.")
                     return None
                bbox = annotation['bbox']
                # Basic check for valid bbox
                if len(bbox) != 4 or not all(isinstance(c, (int, float)) for c in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
                     print(f"Warning: Annotation {anno_id} has invalid 'bbox': {bbox}. Skipping.")
                     return None

                input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                input_boxes = [input_box.tolist()] # processor expects a list of box lists per image/instance

            elif self.prompt_type in ["random_point", "multiple_points"]:
                 # Sample points from the positive mask pixels
                 positive_pixels = np.argwhere(mask > 0) # Returns (row, col) i.e., (y, x)
                 num_total_positive = len(positive_pixels)

                 if num_total_positive == 0:
                      # If mask is empty, we cannot sample points.
                      # Depending on strategy, you might skip, or use a default point (e.g., center),
                      # or maybe fall back to bbox if available.
                      # For simplicity, let's skip this annotation if prompt is points and mask is empty
                      print(f"Warning: Mask is empty for annotation {anno_id}, but prompt_type is '{self.prompt_type}'. Skipping.")
                      return None

                 num_points_to_sample = self.num_points_prompt if self.prompt_type == "multiple_points" else 1
                 # Sample points (y, x) coordinates
                 sampled_indices = np.random.choice(num_total_positive, size=min(num_points_to_sample, num_total_positive), replace=False)
                 sampled_pixels_yx = positive_pixels[sampled_indices]

                 # SAM processor expects points in (x, y) format, as a list of lists (list of point sets)
                 # e.g., [[[x1, y1], [x2, y2]], [[x3, y3]]] where inner list is a point set
                 # We are generating one point set per annotation
                 input_points_xy = sampled_pixels_yx[:, ::-1].tolist() # Convert [y, x] numpy to [x, y] list
                 input_points = [input_points_xy] # processor expects list of point sets per image/instance

            else:
                 # If no prompt type is explicitly handled and the model requires prompts, this will fail later.
                 # SAM models usually require at least one prompt type.
                 print(f"Warning: Unsupported prompt_type '{self.prompt_type}'. No prompts generated.")
                 pass # No prompts will be added to inputs

            # Process the image and mask target using the SAM processor
            # We provide both the image and the ground truth mask
            # The processor will handle resizing, normalization, and preparing mask labels
            # It adds a batch dimension (size 1) for a single image.

            inputs = self.processor(images=image,
                                    input_boxes=input_boxes,   # Pass None if not used
                                    input_points=input_points, # Pass None if not used
                                    masks=[mask], # Provide the ground truth mask (as a list)
                                    return_tensors="pt")

            # The inputs dictionary will contain keys like 'pixel_values',
            # potentially 'input_boxes', 'input_points', and 'mask_labels'.
            # The tensors will have a batch dimension of 1 at dim 0.

            # DO NOT squeeze here. Let default_collate handle stacking,
            # and we will squeeze the excess dimension 1 in the training loop.
            # inputs = {k: v.squeeze(0) if v.ndim == 4 else v.squeeze(0) if v.ndim == 3 and k != 'pixel_values' else v for k, v in inputs.items()} # Removed this line


            return inputs

        except Exception as e:
             print(f"Error processing item {idx} (image_id: {img_id}, anno_id: {anno_id}): {e}. Skipping.")
             # Return None and filter in collate_fn
             return None


# --- Training Function ---

def train_model(model, dataloader, optimizer, device, num_epochs, output_dir):
    model.to(device)
    model.train() # Set model to training mode

    # Freeze image encoder and prompt encoder
    # You can adjust this based on your fine-tuning strategy
    # Find all parameters, identify mask_decoder params
    mask_decoder_param_names = [name for name, _ in model.named_parameters() if "mask_decoder" in name]

    for name, param in model.named_parameters():
         if name in mask_decoder_param_names:
              param.requires_grad = True
              # print(f"Unfrozen parameter: {name}") # Optional: verify what is unfrozen
         else:
              param.requires_grad = False # Freeze other parameters

    # Re-define optimizer after setting requires_grad
    # This ensures only unfrozen parameters are included
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=CONFIG["mask_decoder_learning_rate"], # Use decoder LR as base
                           weight_decay=CONFIG["weight_decay"])
    print(f"Optimizer configured for {len(list(filter(lambda p: p.requires_grad, model.parameters())))} trainable parameters.")


    print(f"Training on device: {device}")

    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        total_loss = 0
        num_batches = 0

        for batch in progress_bar:
            # Handle potential None items if collate_fn was used to filter
            if batch is None:
                 progress_bar.write(f"Skipping empty batch at global step {global_step}.")
                 continue # Skip empty batches

            # Move batch to device
            # The batch dictionary contains tensors stacked by default_collate.
            # Processor output for single item: (1, ...)
            # default_collate stacks these: (batch_size, 1, ...)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Squeeze the unnecessary dimension 1 added by collation
            # This dimension existed because processor outputted (1, ...) for single items
            # Check common keys that might have this extra dim
            keys_to_squeeze = ['pixel_values', 'input_boxes', 'input_points', 'mask_labels']
            for key in keys_to_squeeze:
                if key in batch and batch[key].ndim > 0 and batch[key].size(1) == 1:
                    batch[key] = batch[key].squeeze(1)

            # Now the tensors should have the shapes the model expects:
            # pixel_values: (batch_size, 3, H, W)
            # mask_labels: (batch_size, 1, H_downscaled, W_downscaled)
            # input_boxes: (batch_size, N_boxes, 4) # N_boxes is 1 in your current setup
            # input_points: (batch_size, N_points, 2) # N_points depends on num_points_prompt

            # Forward pass
            # When ground_truth_masks (or mask_labels from processor) are provided,
            # the Hugging Face SamModel directly computes the loss.
            # Pass only the keys the model's forward method accepts
            model_inputs = {}
            if 'pixel_values' in batch: model_inputs['pixel_values'] = batch['pixel_values']
            if 'input_boxes' in batch: model_inputs['input_boxes'] = batch['input_boxes']
            if 'input_points' in batch: model_inputs['input_points'] = batch['input_points']
            if 'mask_labels' in batch: model_inputs['ground_truth_mask'] = batch['mask_labels'] # Use ground_truth_mask for loss computation

            # Verify shapes before passing to model (Optional, for debugging)
            # for k, v in model_inputs.items():
            #     print(f"Batch tensor shape for {k}: {v.shape}")


            # SamModel forward expects 'pixel_values', 'input_boxes', 'input_points',
            # and optionally 'ground_truth_mask' for loss calculation.
            try:
                outputs = model(**model_inputs) # Using **model_inputs unpacks the dictionary as keyword arguments
            except Exception as e:
                 print(f"Error during model forward pass for batch at global step {global_step}: {e}")
                 # print(f"Batch tensor shapes: {[f'{k}: {v.shape}' for k, v in batch.items()]}") # Print shapes for debugging
                 continue # Skip batch on error


            # The loss is typically returned directly by the model when masks are given
            loss = outputs.loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar description with current loss
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        checkpoint_path = os.path.join(output_dir, f"sam_finetuned_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

# --- Main Execution ---

if __name__ == "__main__":
    # Resolve the SAM-2 issue first
    # The checkpoint 'facebook/sam2-hiera-large' does not have the necessary config files for
    # standard from_pretrained loading in transformers.
    # You MUST use a SAM-1 model (like 'facebook/sam-vit-base')
    # OR find a SAM-2 model that IS properly formatted for transformers
    # OR implement manual model/processor loading for 'facebook/sam2-hiera-large'.
    # For this script to run as is, use a SAM-1 model checkpoint name.
    if CONFIG["model_checkpoint"] == "facebook/sam2-hiera-large":
        print("\n---------------------------------------------------------------------------")
        print("IMPORTANT: The checkpoint 'facebook/sam2-hiera-large' is NOT compatible with")
        print("standard SamProcessor/SamModel.from_pretrained loading in transformers.")
        print("Please change CONFIG['model_checkpoint'] to a compatible SAM-1 model")
        print("(e.g., 'facebook/sam-vit-base') or find a compatible SAM-2 checkpoint.")
        print("Alternatively, you must implement manual loading for this model.")
        print("---------------------------------------------------------------------------\n")
        sys.exit(1) # Exit because this checkpoint won't work with the current loading logic

    # Load processor
    # image_size argument helps processor know target size
    processor = SamProcessor.from_pretrained(CONFIG["model_checkpoint"], image_size=CONFIG["image_size"])

    # Create dataset
    train_dataset = CocoSamDataset(
        images_dir=CONFIG["images_dir"],
        annotations_file_path=CONFIG["annotations_file_path"],
        processor=processor,
        prompt_type=CONFIG["prompt_type"],
        num_points_prompt=CONFIG["num_points_prompt"]
    )

    # Custom collate_fn to filter out None items (from dataset errors/skips)
    # default_collate handles stacking tensors
    def collate_fn(batch):
        # Filter out any None items from the batch
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the batch is empty after filtering

        # default_collate stacks lists of tensors into batches
        # It expects each item in the batch list to be a dictionary of tensors
        # where the tensors for a single item do NOT have a batch dimension (size 1 at dim 0)
        # BUT SamProcessor for single items *does* add a batch dim of 1.
        # So default_collate will stack (1, ...) into (batch_size, 1, ...)
        # We will squeeze this excess dim 1 in the training loop.
        return torch.utils.data.dataloader.default_collate(batch)


    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True, # Shuffle for training
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn # Use the custom collate function
    )

    if len(train_dataloader) == 0:
         print("Error: Dataloader is empty. Check dataset path and annotation file.")
         sys.exit(1)


    # Load model
    # Use SamModel for fine-tuning parts.
    model = SamModel.from_pretrained(CONFIG["model_checkpoint"])


    # Optimizer will be defined inside train_model after freezing layers
    optimizer = None # Placeholder

    # Create output directory if it doesn't exist
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Start training
    train_model(model, train_dataloader, optimizer, CONFIG["device"], CONFIG["num_epochs"], CONFIG["output_dir"])

    print("Fine-tuning complete.")