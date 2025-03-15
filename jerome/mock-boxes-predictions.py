import torch
import os
import sys
import cv2
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from scripts.data.augmentations import get_transform
from scripts.models import frcnn
from scripts.config.config import CLASS_CODES_MAP

# Set device
device = torch.device('cpu')
num_classes = 3
THRESHOLD = 0.5  # Confidence threshold for Faster R-CNN

# Load pre-trained Faster R-CNN model
model = frcnn.get_model(num_classes)
model.to(device)
model.load_state_dict(torch.load('saved_models/model_frcnn_32.pth', map_location=torch.device('cpu')))
model.eval()

# Load an image
IMAGE_PATH = 'scripts/data/blueberries/test/sample.png'
image = read_image(IMAGE_PATH)
image_np = cv2.imread(IMAGE_PATH)  # Load in OpenCV format for dimensions

# Image dimensions
height, width, _ = image_np.shape

# Define two fixed square bounding boxes (mock instance segmentation)
box_size = min(width, height) // 4  # Define a size for the square boxes
mock_boxes = torch.tensor([
    [width // 4, height // 4, width // 4 + box_size, height // 4 + box_size],
    [3 * width // 4 - box_size, 3 * height // 4 - box_size, 3 * width // 4, 3 * height // 4]
], dtype=torch.int64)

# Draw mock segmentation boxes
mock_image = draw_bounding_boxes(image, mock_boxes, [f"Mock Bush {i+1}" for i in range(2)], colors="green")

# Convert for OpenCV display
mock_image_np = mock_image.permute(1, 2, 0).numpy()
mock_image_bgr = cv2.cvtColor(mock_image_np, cv2.COLOR_RGB2BGR)

# Function to resize image for display
def resize_for_display(image, max_width=800, max_height=600):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)))

# Resize and display mock segmentation
mock_image_bgr_resized = resize_for_display(mock_image_bgr)
cv2.imshow("Mock Instance Segmentation", mock_image_bgr_resized)
cv2.waitKey(0)

# Run Faster R-CNN on each bush
for i, box in enumerate(mock_boxes):
    x1, y1, x2, y2 = box
    bush_image = image[:, y1:y2, x1:x2]
    with torch.inference_mode():
        eval_transform = get_transform(train=False)
        x = eval_transform(bush_image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

        # Apply thresholding
        pred_boxes = pred["boxes"].long()
        pred_labels = [f"{CLASS_CODES_MAP[label]}: {score:.3f}" for label, score in \
                       zip(pred["labels"], pred["scores"]) if score >= THRESHOLD]

        # Filter based on threshold
        keep = pred["scores"] >= THRESHOLD
        pred_boxes = pred_boxes[keep]

        # Draw Faster R-CNN detections
        output_image = draw_bounding_boxes(bush_image, pred_boxes, pred_labels, colors="blue")

        # Convert and resize for display
        output_image_np = output_image.permute(1, 2, 0).numpy()
        output_image_bgr = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)
        output_image_bgr_resized = resize_for_display(output_image_bgr)
        cv2.imshow(f"Faster R-CNN Detection on Mock Bush {i+1}", output_image_bgr_resized)
        cv2.waitKey(0)

cv2.destroyAllWindows()
