# cyto-mask
## Cytoplasm mask generation using deep learning

Aim: To build a deep learning model to predict cytoplasmic masks from brightfield images

### Data
So far: 500 images clicked; the goal is 1000 images; 50 images have ground truth
Working to get the ground truth for the remaining images

We have processed composite tiff files with 4 channels. 
Channel 1 - nucleus stain
Channel 2 - brightfield 
Channel 3 - mRNA
Channel 4 - cytoplasmic masks from cell pose


### Image preprocessing
Data extraction
The images are a single composite image with tile scans. Each tile scan has 4 channels. These channels should be split. For our project, we will need, 
Input: channel 2 - brightfield image
Output: channel 4 - ground truth with cytoplasmic masks

Augmentation
Depending on how many images I can collect we can decide on this
Geometric transformations (flipping, rotation, scaling).
Intensity normalization (normalize pixel values for contrast variation).
Gaussian blurring or noise addition to make the model robust.

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

Model Evaluation


