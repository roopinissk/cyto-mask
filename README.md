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

