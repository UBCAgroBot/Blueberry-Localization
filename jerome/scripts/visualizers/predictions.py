import torch
import os
import sys

PROJECT_ROOT = '/Users/jeromecho/Library/CloudStorage/OneDrive-Personal/ML/AppliedAI/24-M-14-FPR/jerome/'
sys.path.append(PROJECT_ROOT)

from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from scripts.data.augmentations import get_transform
from scripts.models import frcnn
from scripts.config.config import CLASS_CODES_MAP
import cv2

device = torch.device('cpu')
num_classes = 3

# load model
model = frcnn.get_model(num_classes)
model.to(device)
model.load_state_dict(torch.load('saved_models/model_frcnn_12.pth', map_location=torch.device('cpu')))
model.eval()

IMAGE_DIR = 'data/blueberries/test'

image_and_xml_paths = os.listdir(IMAGE_DIR)
image_paths = []
for image_path in image_and_xml_paths:
    if image_path.endswith('jpg'):
        image_paths.append(IMAGE_DIR + '/' + image_path)

with torch.inference_mode():
    for image_path in image_paths:
        image = read_image(image_path)
        eval_transform = get_transform(train=False)

        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

        # (image - image.min()) / (image.max() - image.min()) regularizies all 
        # values of the image to between 0 and 1 inclusive (max == 1, min == 0)
        # Multiplying by 255 gives us a newly generated image whose pixel values 
        # range from 0 to 255 (which is conventional for many image formats)
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)

        pred_boxes = pred["boxes"].long()
        pred_labels = [f"{CLASS_CODES_MAP[label]}: {score: .3f}" for label, score in \
                    zip(pred["labels"], pred["scores"])]
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
        output_image_pmt = output_image.permute((1, 2, 0))
        output_image_np = output_image_pmt.numpy()
        output_image_bgr = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)

        cv2.imshow("Result", output_image_bgr)
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()


