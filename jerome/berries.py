import io
import os
import pandas as pd
from PIL import Image
from openpyxl import load_workbook
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from scripts.models import frcnn
from scripts.data.augmentations import get_transform

# Configuration
EXCEL_PATH = os.path.abspath('Blueberries.xlsx')
IMAGE_OUTPUT_DIR = 'images_blue/final attempt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Extract images and counts from Excel ---
wb = load_workbook(EXCEL_PATH, data_only=True)
ws = wb['Sheet1']
df = pd.read_excel(EXCEL_PATH)

# Extract embedded images
images = ws._images
image_data = []
for img in images:
    img_bytes = io.BytesIO(img._data())
    pil = Image.open(img_bytes)
    image_data.append(pil)

# Berry counts
berry_counts = df['Blueberries'].dropna().tolist()
image_berry_pairs = list(zip(image_data, berry_counts))

# --- 2. Preprocess images: convert red berries to natural blue ---
def replace_red_with_natural_blue(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l[mask>0] = np.clip(l[mask>0]*0.4, 0, 255)
    a[mask>0] = np.clip(a[mask>0] - 80, 0, 255)
    b[mask>0] = np.clip(b[mask>0] + 80, 0, 255)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# Apply conversion and save
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
for idx, (pil_img, count) in enumerate(image_berry_pairs):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    converted = replace_red_with_natural_blue(img_bgr)
    out_path = os.path.join(IMAGE_OUTPUT_DIR, f'temp_image_{idx}.jpg')
    cv2.imwrite(out_path, converted)

# --- 3. (Optional) Visualize few processed images ---
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes = axes.flatten()
for i, (pil_img, _) in enumerate(image_berry_pairs[:9]):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = replace_red_with_natural_blue(img)
    axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[i].axis('off')
plt.tight_layout()
plt.show()

print('Processed and saved all images to', IMAGE_OUTPUT_DIR)
