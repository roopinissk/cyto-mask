# cyto-mask
## Cytoplasm mask generation using deep learning

Aim: Predicting cytoplasmic masks from bright field microscopy images.
Methods: Train, hyperparameter tune, and compare various NN (U-net, SAM2 (transformer-based), microSAM (vit_l_lm), and Mask Region-based Convolutional Neural Network).

### Data
Images were collected from HCT116 colorectal cancer cell line
540 original images clicked augmented to 1000
Augmentation method applied - Rotation

Dataset can be found [here](https://www.kaggle.com/datasets/rubyssk/cytomask-microscopy-images)

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

U-Net: A deep learning architecture for semantic segmentation that uses an encoder-decoder structure with skip connections to capture spatial hierarchies in images, particularly effective in medical image segmentation.
Tried inferencing through the object based detection; Gives a satisfactory output so far, planning to train the model and check the results.

SAM2 (Transformer-based): A variant of the SAM (Segment Anything Model), leveraging transformer-based architectures for efficient and high-quality image segmentation, especially in complex and varied datasets.

microSAM (vit_l_lm): A micro version of SAM that utilizes Vision Transformer (ViT) with a large model (ViT_L) for improved segmentation performance in smaller and more detailed images, optimizing efficiency and accuracy.

Mask Region-based Convolutional Neural Network (Mask R-CNN): An extension of Faster R-CNN that not only detects objects in an image but also generates pixel-wise segmentation masks for each detected object, enabling more precise object localization.



Model architecture experiments
Input - greyscale image; maybe we can experiment with different channels
Model architecture  - whatever model we design
Output processing - selection of activation

Training
Loss
Optimizer
Batches and epochs

## Collaborators 
Ruby Kumar - Data collection, image pre-processing, SAM2 model training and evaluation

Jonathan Amsalem - Image pre-processing, MicroSAM model training and evaluation

Divya Rallapalli - Unet model training and evaluation

Zak Aminzada - Mask R-CNN model training and evaluation


## Directories

Models - code to instantiate, explore, train, hyperparameter tune, and test models.

Config - config files used to map augmented images to original ones and configuration for png to rle format processing. 

Preprocess - code to extract images from tiff format, clean images, generate masks and contours for ground truth. 

