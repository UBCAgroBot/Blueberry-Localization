import os
import torch 
import torchvision
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

class YOLOTxtDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 
            os.path.exists(os.path.join(annotation_dir, f.rsplit('.', 1)[0] + ".txt"))
        ])
        self.transforms = transforms
        self.resize_to = (640, 640)  # height, width

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        txt_path = os.path.join(self.annotation_dir, img_name.rsplit('.', 1)[0] + ".txt")

        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        image = F.resize(image, self.resize_to)
        img_tensor = F.to_tensor(image)
        new_width, new_height = image.size

        # Scale factors
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        # Load YOLO-style annotation and rescale boxes
        boxes = []
        labels = []
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_center, y_center, w, h = map(float, parts)
                        x1 = (x_center - w / 2) * orig_width * scale_x
                        y1 = (y_center - h / 2) * orig_height * scale_y
                        x2 = (x_center + w / 2) * orig_width * scale_x
                        y2 = (y_center + h / 2) * orig_height * scale_y
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls) + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target

class BlueberryDetector:
    def __init__(self, num_classes=2, pretrained=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # Initialize model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    def train(self, train_loader, val_loader, num_epochs=10, lr=0.005):
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            skipped = 0

            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = [img.to(self.device) for img in images]
                
                # Filter out boxes with zero area
                cleaned_targets = []
                for t in targets:
                    boxes = t["boxes"]
                    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    if keep.sum() == 0:
                        continue
                    t_cleaned = {
                        "boxes": boxes[keep].to(self.device),
                        "labels": t["labels"][keep].to(self.device),
                        "image_id": t["image_id"].to(self.device),
                    }
                    cleaned_targets.append(t_cleaned)

                if len(cleaned_targets) != len(images):
                    skipped += 1
                    continue

                try:
                    loss_dict = self.model(images, cleaned_targets)
                    losses = sum(loss for loss in loss_dict.values())

                    if torch.isnan(losses):
                        print("[Warning] NaN loss detected, skipping batch.")
                        skipped += 1
                        continue

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    epoch_loss += losses.item()

                except Exception as e:
                    print(f"[Error] Exception during training step: {e}")
                    skipped += 1

            lr_scheduler.step()
            avg_loss = epoch_loss / max(1, len(train_loader) - skipped)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f} | Skipped: {skipped}")

    def evaluate(self, val_loader, iou_threshold=0.5, conf_threshold=0.5):
        self.model.eval()
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for images, targets in val_loader:
            images = [img.to(self.device) for img in images]
            gt_boxes = [t["boxes"].to(self.device) for t in targets]

            with torch.no_grad():
                predictions = self.model(images)

            for pred, gt_box in zip(predictions, gt_boxes):
                pred_boxes = pred["boxes"][pred["scores"] > conf_threshold]

                if len(pred_boxes) == 0:
                    false_negatives += len(gt_box)
                    continue

                iou_matrix = box_iou(pred_boxes, gt_box)
                matches = iou_matrix > iou_threshold

                matched_gt = torch.any(matches, dim=0)
                matched_preds = torch.any(matches, dim=1)

                true_positives += matched_gt.sum().item()
                false_negatives += (~matched_gt).sum().item()
                false_positives += (~matched_preds).sum().item()

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def visualize_predictions(self, val_loader, num_images=6, conf_thresh=0.3):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        self.model.eval()
        for i, (img, target) in enumerate(val_loader):
            if i >= num_images:
                break
                
            img = img[0].to(self.device)
            img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
            gt_boxes = target[0]['boxes']
            
            with torch.no_grad():
                pred = self.model([img])[0]
            
            # Filter predictions by confidence
            scores = pred['scores'].cpu()
            keep = scores > conf_thresh
            pred_boxes = pred['boxes'][keep].cpu()
            
            # Plot side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(img_np)
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1, 
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                ax1.add_patch(rect)
            ax1.set_title(f"Ground Truth (Image {i+1})")
            ax1.axis("off")
            
            ax2.imshow(img_np)
            for box in pred_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax2.add_patch(rect)
            ax2.set_title(f"Prediction (Image {i+1})")
            ax2.axis("off")
            plt.tight_layout()
            plt.show()

def create_data_loaders(image_dir, annotation_dir, batch_size=2, train_ratio=20/26):
    dataset = YOLOTxtDataset(image_dir, annotation_dir)
    
    # Split dataset
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    return train_loader, val_loader

# Initialize and train
detector = BlueberryDetector(num_classes=2)
train_loader, val_loader = create_data_loaders(
    image_dir="images_blue/final attempt",
    annotation_dir="annotations"
)

# Train model
detector.train(train_loader, val_loader)

# Visualize predictions
detector.visualize_predictions(val_loader)

# Evaluate metrics
metrics = detector.evaluate(val_loader)
print(metrics)

# Save model
detector.save_model("saved_models/model_finetuned.pth")