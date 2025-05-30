import os
import numpy as np
import tifffile as tiff
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create output directories if they don't exist
os.makedirs('./dataset/bf', exist_ok=True)
os.makedirs('./dataset/mask', exist_ok=True)
os.makedirs('./dataset/binary_masks', exist_ok=True)

# Read the TIFF file
file = "Composite_with_masks.tif"
img = tiff.imread(file)
print(f"Original image shape: {img.shape}")

# Extract brightfield and mask channels
bf = img[:, 1, :, :]  # Channel 1 for brightfield
mask = img[:, 3, :, :]  # Channel 3 for mask

print(f"Brightfield shape: {bf.shape}")
print(f"Mask shape: {mask.shape}")

# Save individual brightfield and mask images
for i in tqdm(range(0, bf.shape[0]), desc="Saving brightfield and mask images"):
    # Save brightfield image
    bfi = bf[i]
    io.imsave(f'./dataset/bf/{i}.png', bfi)

    
    # Save mask image
    maski = mask[i].astype(np.uint8)
    io.imsave(f'./dataset/mask/{i}.png', maski)
    
    # Create directory for binary masks for this frame
    frame_dir = f'./dataset/binary_masks/{i}'
    os.makedirs(frame_dir, exist_ok=True)
    
    # Get unique values in this mask
    unique_values = np.unique(maski)
    print(f"Frame {i} has {len(unique_values)} unique values: {unique_values}")
    
    # Create binary masks for each unique value (except 0, which is typically background)
    mask_count = 0
    for color in unique_values:
        if color == 0:  # Skip background
            continue
            
        # Create binary mask where the pixels with the color are 1 and the rest are 0
        binary_mask = np.where(maski == color, 1, 0)
        
        # Save the binary mask (multiplied by 255 for better visualization)
        io.imsave(f'{frame_dir}/{color}.png', binary_mask.astype(np.uint8) * 255)
        mask_count += 1
    
    print(f"Created {mask_count} binary masks for frame {i}")


# # Visualization example for the first frame
# if mask.shape[0] > 0:
#     # Get the first mask
#     mask_0 = mask[0].astype(np.uint8)
#     unique_colors = np.unique(mask_0)
    
#     # Create a figure to visualize original and binary masks
#     n_colors = len(unique_colors)
#     fig, axes = plt.subplots(1, n_colors + 1, figsize=(5 * (n_colors + 1), 5))
    
#     # Plot the original mask
#     axes[0].imshow(mask_0, cmap='nipy_spectral')
#     axes[0].set_title('Original Mask')
#     axes[0].axis('off')
    
#     # Plot binary masks
#     for idx, color in enumerate(unique_colors):
#         if color == 0 and len(unique_colors) > 1:  # Skip background if it's not the only value
#             continue
            
#         binary_mask = np.where(mask_0 == color, 1, 0)
#         ax_idx = idx + 1 if color != 0 else 1
#         if ax_idx < len(axes):
#             axes[ax_idx].imshow(binary_mask, cmap='gray')
#             axes[ax_idx].set_title(f'Binary Mask: Color {color}')
#             axes[ax_idx].axis('off')
    
#     plt.tight_layout()
#     plt.savefig('./dataset/visualization.png')
#     plt.close()

# print("Processing complete!")