import os
import numpy as np
import tifffile as tiff
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image

try:
    with Image.open("/home/suriya/cyto-mask/all-corrected_final_merge.tif") as img:
        img.verify()  # Verify that it is, indeed, an image
    print("TIFF file is valid and not corrupt.")
except (IOError, SyntaxError) as e:
    print("TIFF file is corrupt or unreadable:", e)