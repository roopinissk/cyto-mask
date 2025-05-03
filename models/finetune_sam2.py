import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm
import random

# Assuming you have Hugging Face transformers installed
from transformers import SamModel, SamProcessor # Or potentially a Sam2Model/Sam2Processor
# Import the specific output type if needed for type checking (optional)
# from transformers.models.sam.modeling_sam import SamMaskDecoderOutput # This is the expected output with loss

# Assuming pycocotools is installed for RLE decoding
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO # Useful for loading annotations

# --- Configuration and Hyperparameters ---
CONFIG = {
    "images_dir": "50_dataset/bf",  # Directory containing your images
    "annotations_file_path": "individual_rle_annotations.json", # Your COCO RLE JSON file
    "model_checkpoint": "facebook/sam-vit-base", # SAM model checkpoint (replace with SAM-2 if available)
    "output_dir": "sam_finetuned_checkpoint", # Directory to save checkpoints

    "num_epochs": 10,
    "batch_size": 8, # Adjust based on GPU memory
    "learning_rate": 1e-5, # Learning rate for fine-tuning (usually small) - Note: not directly used if optimizer is defined with mask_decoder_learning_rate
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
        # Filter annotations to ensure they have segmentation (RLE format) and bbox, and positive area
        all_anno_ids = self.coco.getAnnIds(imgIds=self.img_ids)
        annotations = self.coco.loadAnns(all_anno_ids)

        # Pre-filter annotations
        for annotation in annotations:
            img_id = annotation['image_id']
            anno_id = annotation['id']
            # Check for RLE segmentation specifically (dict type)
            has_segmentation_rle = 'segmentation' in annotation and isinstance(annotation['segmentation'], dict)
            has_bbox = 'bbox' in annotation
            positive_area = annotation.get('area', 0) > 0 # Use .get for safety

            # Filter criteria: Must have RLE segmentation, bbox, and positive area
            # Bbox is needed for 'bbox' prompt, Seg+Area needed for ground truth mask and point prompts
            if has_segmentation_rle and has_bbox and positive_area:
                 self.annotation_pairs.append((img_id, anno_id))
            else:
                 # Optional: print skipped annotations and reason
                 # print(f"Skipping anno {anno_id} (img {img_id}): Missing RLE seg ({has_segmentation_rle}), bbox ({has_bbox}), or area ({positive_area}).")
                 pass


        print(f"Loaded {len(self.annotation_pairs)} valid annotations for training across {len(self.img_ids)} images.")

    def __len__(self):
        return len(self.annotation_pairs)

    def __getitem__(self, idx):
        img_id, anno_id = self.annotation_pairs[idx]

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.images_dir, img_info['file_name'])
        try:
            image = Image.open(image_path).convert("RGB") # Ensure RGB
        except FileNotFoundError:
             print(f"Error: Image file not found at {image_path} for anno {anno_id}. Skipping.")
             return None
        except Exception as e:
             print(f"Error loading image {image_path} for anno {anno_id}: {e}. Skipping.")
             return None


        # Load annotation (already filtered, but load again for safety)
        annotation = self.coco.loadAnns(anno_id)[0]

        # Decode RLE segmentation mask
        # We already filtered for RLE format and valid area
        if 'segmentation' in annotation and isinstance(annotation['segmentation'], dict):
            try:
                mask = self.coco.annToMask(annotation) # pycocotools decodes RLE for us
                # Ensure mask is binary uint8
                mask = (mask > 0).astype(np.uint8)
                 # Double check mask area after decoding, just in case annToMask produced empty for valid RLE
                if np.sum(mask) == 0:
                     print(f"Warning: Decoded mask is empty for anno_id {anno_id} (img_id {img_id}). Skipping.")
                     return None
            except Exception as e:
                 print(f"Error decoding mask for anno_id {anno_id} (img_id {img_id}): {e}. Skipping.")
                 return None
        else:
            # This should not happen with the filtering, but as a safeguard
             print(f"Annotation {anno_id} (img_id {img_id}) missing RLE seg after filtering. Skipping.")
             return None


        # Generate Prompt (using bbox or points derived from the mask)
        input_boxes = None
        input_points = None
        input_labels = None # Labels for points (1 for foreground, 0 for background)

        if self.prompt_type == "bbox":
            # Bbox format [x, y, w, h], convert to [x1, y1, x2, y2]
            bbox = annotation['bbox']
            # Convert [x, y, w, h] to [x1, y1, x2, y2] floats
            x1, y1, w, h = bbox
            input_box_coords = [float(x1), float(y1), float(x1 + w), float(y1 + h)]
            # Processor expects list of list of floats for input_boxes per image/item
            # Example: input_boxes=[[[x1, y1, x2, y2]]] for one box
            input_boxes = [[input_box_coords]]


        elif self.prompt_type in ["random_point", "multiple_points"]:
             # Sample points from the positive mask pixels
             positive_pixels = np.argwhere(mask > 0) # Returns (row, col) i.e., (y, x)
             # We already filtered for positive area, so positive_pixels should not be empty
             if len(positive_pixels) == 0:
                  print(f"Warning: No positive pixels found in mask for anno_id {anno_id}. Skipping point prompt.")
                  return None

             # Sample points (y, x)
             num_points_to_sample = self.num_points_prompt if self.prompt_type == "multiple_points" else 1
             # Ensure we don't sample more points than available positive pixels
             num_points_to_sample = min(num_points_to_sample, len(positive_pixels))

             sampled_indices = np.random.choice(len(positive_pixels), size=num_points_to_sample, replace=False)
             # Convert (y, x) points to (x, y) lists of floats for the processor
             input_points_xy = [[float(p[1]), float(p[0])] for p in positive_pixels[sampled_indices]]
             # Processor expects list of list of floats for input_points per image/item
             # Example: input_points=[[[x1, y1], [x2, y2]]] for two points
             input_points = [input_points_xy]
             # Labels for points (1 for foreground, 0 for background). Processor expects list of list of ints/longs.
             # Since we sample from positive pixels, all labels are 1.
             input_labels = [[1] * num_points_to_sample] # Example: input_labels=[[1, 1]]


        else:
            # If no prompt type is specified, SAM cannot segment.
            print(f"Error: Unsupported or missing prompt_type '{self.prompt_type}' for anno {anno_id}. Skipping.")
            return None

        # Process the image, prompts, AND ground truth masks using the SAM processor
        # Use the 'ground_truth_masks' argument as per processor documentation
        # Pass inputs as lists, even for a single item, so processor expects a batch dimension (dim=1).
        try:
            processor_inputs = self.processor(
                images=[image], # <-- Pass as list of PIL Images
                input_boxes=input_boxes, # <-- Pass as list of list of list of floats (or None)
                input_points=input_points, # <-- Pass as list of list of list of floats (or None)
                input_labels=input_labels, # <-- Pass as list of list of ints/longs (or None)
                ground_truth_masks=[mask], # <-- Pass the ground truth mask as a list of numpy arrays
                return_tensors="pt"
            )

        except Exception as e:
             print(f"Error processing item {idx} (image_id: {img_id}, anno_id: {anno_id}) with processor: {e}. Skipping.")
             return None

        # The processor should return a dictionary with keys like
        # 'pixel_values', 'input_boxes', 'input_points', 'input_labels', 'mask_labels'.
        # 'mask_labels' should be the downscaled ground truth masks ready for loss.
        # These tensors will have a batch dimension of 1.
        # default_collate will stack these dim=1 tensors correctly into a batch.

        # Check if mask_labels was actually generated by the processor
        # This key *must* be present and valid for model loss computation
        if 'mask_labels' not in processor_inputs or processor_inputs['mask_labels'] is None:
             print(f"Warning: Processor failed to generate 'mask_labels' for item {idx} (img_id {img_id}, anno_id {anno_id}). Skipping.")
             # This indicates an issue with the processor or input mask.
             # For robustness, skip this item if the crucial mask_labels output is missing.
             return None

        # Return the dictionary directly from the processor.
        # default_collate will handle batching items returned this way.
        return processor_inputs

# --- Training Function ---

def train_model(model, dataloader, optimizer, device, num_epochs, output_dir):
    model.to(device)
    model.train() # Set model to training mode

    # --- Explicitly set requires_grad for different parts ---
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze mask decoder parameters
    # Parameters belonging to the mask_decoder will have 'mask_decoder' in their name
    # Other parts (image_encoder, prompt_encoder) will remain frozen
    for name, param in model.named_parameters():
        if "mask_decoder" in name:
            param.requires_grad = True
            # print(f"Unfrozen parameter: {name}") # Optional: verify what is unfrozen
        # Note: Image encoder and prompt encoder parameters will have requires_grad=False
        # as they were set to False initially and don't match the "mask_decoder" condition.

    print(f"Training on device: {device}")
    print("Parameters being trained (mask_decoder only):")
    trainable_param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name}")
            trainable_param_count += 1
    if trainable_param_count == 0:
        print("Warning: No parameters are set to require gradients!")


    global_step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        total_loss = 0
        step_count = 0 # Count valid steps to calculate average loss correctly

        for i, batch in enumerate(progress_bar):
            # Handle potential None items if collate_fn was used to filter
            if batch is None:
                 continue # Skip invalid batches

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # --- Debug: Print batch keys and shapes ---
            # You can limit this to the first few batches or randomly selected ones
            # if i < 3: # Print for first 3 batches
            #     print(f"\n--- Batch {i} Keys and Shapes ---")
            #     for k, v in batch.items():
            #         if isinstance(v, torch.Tensor):
            #             print(f"  {k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")
            #         elif isinstance(v, list):
            #              print(f"  {k}: list with {len(v)} items (e.g., {type(v[0]) if v else 'empty'})")
            #         else:
            #             print(f"  {k}: {type(v)}")
            #     print("-" * 20)
            # --- End Debug ---


            # Forward pass
            # The Hugging Face SamModel is designed to compute loss if ground_truth_mask (i.e., mask_labels) is provided
            # The batch dictionary keys should match what the model expects,
            # typically matching the processor's output keys.
            try:
                 outputs = model(**batch) # Using **batch unpacks the dictionary as keyword arguments
            except Exception as e:
                 print(f"\nError during model forward pass for batch {i}: {e}. Batch keys: {batch.keys()}. Skipping batch.")
                 continue # Skip this batch if the forward pass fails


            # The loss should be returned directly by the model when mask_labels are given
            # Check if the loss attribute exists
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                 # This is the scenario causing the AttributeError, but now we check explicitly
                 print(f"\nWarning: Model output object for batch {i} does not have a 'loss' attribute or loss is None.")
                 print(f"Output object type: {type(outputs)}")
                 # print(f"Output object attributes: {dir(outputs)}") # Uncomment to see all attributes
                 # print(f"Batch keys passed to model: {batch.keys()}") # Uncomment to see keys passed
                 # If loss is critical for every batch, you might want to stop or raise error here
                 # For robustness, we will just skip this batch as no loss can be computed.
                 continue # Skip this batch


            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1
            avg_loss_so_far = total_loss / step_count if step_count > 0 else 0

            # Update progress bar description with current loss
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": avg_loss_so_far})

        # Calculate average loss for the epoch
        avg_loss = total_loss / step_count if step_count > 0 else 0 # Use step_count for average
        print(f"\nEpoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        checkpoint_path = os.path.join(output_dir, f"sam_finetuned_epoch_{epoch+1}.pth")
        # Save only the state_dict for easier loading later
        # Ensure only trainable parameters are saved if you want a smaller checkpoint
        # Or save the whole state_dict and load carefully later
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

# --- Main Execution ---

if __name__ == "__main__":
    # Load processor
    # image_size argument ensures correct input size if model config doesn't match
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
    # and use default collate for the rest
    def collate_fn(batch):
        # Filter out any None items from the batch
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the batch is empty after filtering

        # Use default collate to combine the dictionaries into a batch dictionary
        # default_collate works correctly when __getitem__ returns dictionaries
        # of tensors (or lists of tensors/values).
        return torch.utils.data.dataloader.default_collate(batch)


    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True, # Shuffle for training
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn # Use the custom collate function
    )

    # Load model
    model = SamModel.from_pretrained(CONFIG["model_checkpoint"])

    # Define optimizer
    # Optimizer should be defined *after* setting requires_grad, as it only
    # registers parameters that currently require gradients.
    # We set requires_grad in train_model, but filter here anyway for clarity
    # that we intend to optimize only trainable parameters.
    # This filter list will be empty *before* train_model is called, but
    # the optimizer will update parameters based on their requires_grad state
    # at the time of .step().
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=CONFIG["mask_decoder_learning_rate"], # Use decoder LR
                           weight_decay=CONFIG["weight_decay"])

    # Create output directory if it doesn't exist
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Start training
    train_model(model, train_dataloader, optimizer, CONFIG["device"], CONFIG["num_epochs"], CONFIG["output_dir"])

    print("Fine-tuning complete.")