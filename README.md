# cyto-mask
## Cytoplasm mask generation using deep learning

Aim: To build a deep learning model to predict cytoplasmic masks from brightfield images

### Data
Images were collected from HCT116 colorectal cancer cell line
550 original images clicked augmented to 1000
Augmentation method applied - Rotation

We tried gaussian blur
reducing noise and adding noise

But, these techniques did not work as the images we worked with in itself were very noisy so we stuck to just rotations

### Image preprocessing
Data extraction
The images are a single composite image with tile scans. Each tile scan has 4 channels. These channels should be split. For our project, we will need, 
Input: channel 2 - brightfield image
Output: channel 4 - ground truth with cytoplasmic masks - generated using cellpose
Note: these channel number maybe changes in different tiff files

We first converted the grayscale mask (ground truth) images to binary images 0(white)- cells 1(black)- background

The cells in a field are clumped together and they appear stuck togther to form a cluster, The binary images turned out as whole white clusters, we then isolated each binary mask, got their edges and then merged all the edges to get a clear distinction between the cells. 

Splitting data
Training 80%
Validation 10%
Testing 10%

### Selection of Model
I want to try and replicate this paper. They use Cyto R-CNN, a new architecture for whole-cell segmentation. Their dataset used H&E stained images (histopath slides)

Other architectures to explore - ResNet, UNet (transformer-based)

Model architecture experiments
Input - greyscale image; maybe we can experiment with different channels
Model architecture  - whatever model we design
Output processing - selection of activation

Training
Loss
Optimizer
Batches and epochs

### Other models explored
1. Diffusion based models 
2. MicroSAM
3. SAM2 - Tried inferencing through the object based detection; Gives a satisfactory output so far, planning to train the model and check the results.
4. Unet - a pretrained unet on generic dataset was tried, This basically just gave a black image with a white line as an output



## SAM2

Using Vanilla SAM2 with 


![sam2](./output/sam2-large-segments.png)

Determined mask shape from predictions: (2048, 2048)
Combined predicted mask created. Total predicted foreground pixels: 4166871
Found 25 potential ground truth mask files.
Combined ground truth mask created from 25 files. Total GT foreground pixels: 1178254

Calculating Metrics:
True Positives (TP): 1171819
False Positives (FP): 2995052
False Negatives (FN): 6435

--- Evaluation Metrics ---
Precision: 0.2812
Recall:    0.9945
F2 Score:  0.6598
IoU Score: 0.2808
Calculated Metrics:
{'precision': 0.28122276883534886, 'recall': 0.9945385290429783, 'f2_score': 0.6598160804895552, 'iou_score': 0.28078913935372085, 'TP': 1171819.0, 'FP': 2995052.0, 'FN': 6435.0}


SAM2 also offers more finegrained cofigurations for the box size

which can be defined like this

```
mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,

    # --- Sampling Density & Strategy ---
    points_per_side=64,          # Keep high: Need dense points to find faint objects.
    points_per_batch=128,        # Keep: Fine, relates to GPU memory/speed.
    crop_n_layers=1,             # Keep (or even try 2): Essential for low contrast & potentially small features. Cropping helps focus.
    crop_n_points_downscale_factor=1, # Changed from 2: Use full point density within crops for better detail on faint objects, accept speed trade-off.

    # --- Quality/Filtering Thresholds (Key Tuning Parameters) ---
    pred_iou_thresh=0.80,        # Increased from 0.7: Let's be slightly more selective initially. Low contrast might make SAM less confident, so start lower than default (0.86) but higher than your previous 0.7. *Adjust this down if cells are missed.*
    stability_score_thresh=0.90, # Decreased from 0.92: Lowering slightly as perfect stability might be hard with faint edges. Try to capture less 'stable' but real objects. *Adjust this up/down based on noise vs. missed cells.*
    stability_score_offset=1.0,  # Reset to default: Less critical to tune initially.

    # --- Postprocessing ---
    box_nms_thresh=0.7,          # Keep: Standard value for removing highly redundant duplicates.
    min_mask_region_area=100,    # Increased significantly from 25: Crucial for ignoring noise/debris. Estimate the pixel area of the smallest *real* cell you care about and set slightly below that. 100 is a guess - **adjust based on your image resolution and cell size.**

    # --- Model Features ---
    use_m2m=True                 # Keep: Usually improves mask quality.
)
```

![sam2-fine](./output/sam2-fine-grained.png)


Determined mask shape from predictions: (2048, 2048)
Combined predicted mask created. Total predicted foreground pixels: 4179406
Found 25 potential ground truth mask files.
Combined ground truth mask created from 25 files. Total GT foreground pixels: 1178254

Calculating Metrics:
True Positives (TP): 1175459
False Positives (FP): 3003947
False Negatives (FN): 2795

--- Evaluation Metrics ---
Precision: 0.2813
Recall:    0.9976
F2 Score:  0.6609
IoU Score: 0.2811
Calculated Metrics:
{'precision': 0.281250254222662, 'recall': 0.9976278459474802, 'f2_score': 0.6609326717875652, 'iou_score': 0.28106229231921637, 'TP': 1175459.0, 'FP': 3003947.0, 'FN': 2795.0}