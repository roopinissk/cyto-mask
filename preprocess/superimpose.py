'''
To check if the bf images and the masks augmented correctly, we can try superimpose the images and the masks.

This can be done by using the following code:
'''

from PIL import Image
import os 
import glob
import numpy as np
import matplotlib.pyplot as plt

def superimpose(bf_dir, gt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    bf_images = sorted(glob.glob(os.path.join(bf_dir, '*.png')))
    gt_images = sorted(glob.glob(os.path.join(gt_dir, '*.png')))


    for bf_path, gt_path in zip(bf_images, gt_images):
        bf_img = Image.open(bf_path)
        gt_img = Image.open(gt_path)

        # print(f"bf_img mode: {bf_img.mode}")
        # print(f"gt_img mode: {gt_img.mode}")

        # Convert both images to RGB mode
        bf_img = bf_img.convert("RGB")
        gt_img = gt_img.convert("RGB")

        # Now to superimpose
        superimposed = Image.blend(bf_img, gt_img, alpha=0.5)
        base_name = os.path.basename(bf_path)
        superimposed.save(os.path.join(output_dir, base_name))
        print(f"saved superimposed image for {base_name}")


# Call the function
bf_augmented_dir = "dataset/augment"
gt_augmented_dir = "dataset/augment_ground_truth"
output_superimposed_dir = "dataset/superimposed"

superimpose(bf_augmented_dir, gt_augmented_dir, output_superimposed_dir)