import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np
import json
from tqdm import tqdm

# --- Dataset ---
class CocoValDataset(Dataset):
    def __init__(self, img_dir, ann_path):
        self.img_dir = img_dir
        self.coco = COCO(ann_path)
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)
        return img, img_id

# --- CONFIGURATION ---
IMG_DIR = '/home/suriya/cyto-mask/50_dataset/bf'
ANN_PATH = '/home/suriya/cyto-mask/val_annotations.json'
MODEL_PATH = '/home/suriya/cyto-mask/Zak/gptmaskrcnn_trained.pth'
BATCH_SIZE = 2
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET & LOADER ---
dataset = CocoValDataset(IMG_DIR, ANN_PATH)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL ---
model = maskrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

# --- Inference & Result Formatting ---
results = []
print("Running validation...")

with torch.no_grad():
    for images, img_ids in tqdm(data_loader):
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = img_ids[i]
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            masks = output['masks'].cpu().numpy()  # shape: (N, 1, H, W)

            for j in range(len(boxes)):
                mask = masks[j, 0]
                mask = (mask > 0.5).astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")  # for JSON serialization

                result = {
                    "image_id": int(image_id),
                    "category_id": int(labels[j]),
                    "bbox": [float(x) for x in boxes[j]],
                    "score": float(scores[j]),
                    "segmentation": rle,
                }
                results.append(result)

# --- SAVE RESULTS JSON ---
results_path = "val_results.json"
with open(results_path, "w") as f:
    json.dump(results, f)

# --- COCO EVALUATION ---
coco_gt = COCO(ANN_PATH)
coco_dt = coco_gt.loadRes(results_path)
coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
