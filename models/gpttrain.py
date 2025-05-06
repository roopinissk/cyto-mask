import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import random

print("train.py is running")

class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_path, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_path)
        self.img_ids = self.coco.getImgIds()
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

            seg = ann['segmentation']
            if isinstance(seg, list):  # polygon
                mask = self.coco.annToMask(ann)
            else:
                mask = maskUtils.decode(seg)
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# --- CONFIGURATION ---
IMG_DIR = '/home/suriya/cyto-mask/50_dataset/bf'
ANN_PATH = '/home/suriya/cyto-mask/train_annotations.json'
BATCH_SIZE = 2
NUM_CLASSES = 2  # 1 foreground class + background
NUM_EPOCHS = 600
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATA AUGMENTATION ---
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# --- DATASET & DATALOADER ---
dataset = CocoDataset(IMG_DIR, ANN_PATH, transforms=train_transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

# --- MODEL ---
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)

# Replace the box predictor for correct number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

# Replace the mask predictor too
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)

model.to(DEVICE)

# --- OPTIMIZER & SCHEDULER ---
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# --- TRAINING LOOP ---
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Print individual losses
        loss_items = {k: v.item() for k, v in loss_dict.items()}
        print(f"[Epoch {epoch+1}] Loss breakdown: {loss_items}")

        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        epoch_loss += losses.item()

    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Loss: {avg_loss:.4f}")

    lr_scheduler.step(avg_loss)

# --- SAVE MODEL ---
print("Training complete. Saving model.")
torch.save(model.state_dict(), "gptmaskrcnn_trained.pth")
