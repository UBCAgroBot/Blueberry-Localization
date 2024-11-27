import torch
import os
from scripts.config.config import CLASSES
from torch.utils.data import Dataset
from torchvision.io import read_image
import xml.etree.ElementTree as ET

class XMLDatasetPyTorch(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        images_and_xml = list(sorted(os.listdir(image_dir)))[:2]
        
        self.images = []
        self.xml = []
        # This logic is too brittle
        for idx in range(0, len(images_and_xml), 2):
          self.images.append(images_and_xml[idx])
          self.xml.append(images_and_xml[idx+1])

    def __len__(self):
        return len(self.xml)

    def __getitem__(self, idx):
        img_path = self.image_dir + '/' + self.images[idx]
        xml_path = self.image_dir + '/' + self.xml[idx]

        image = read_image(img_path)
        annotations = self.parse_xml(xml_path)

        if self.transforms:
          image = self.transforms(image)
        
        target = {}

        target["boxes"] = torch.tensor(annotations['boxes'], dtype=torch.float32)

        area = 0

        # guard against possibility of there being no bounding boxes in the XML
        if len(target["boxes"]) > 0:
          area = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])

        # set iscrowd for all instances equal to 0. Setting instance's iscrowd to 1 ignores 
        # it during evaluation (which we don't want to do)
        iscrowd = torch.zeros((len(annotations["labels"])), dtype=torch.int64)

        target["labels"] = torch.tensor(annotations['labels'], dtype=torch.int64)
        target["image_id"] = idx
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return image, target

    def parse_xml(self, xml_file_path):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        annotations = { "boxes": [], "labels": [] }
        filename = root.find('filename').text
        annotations["filename"] = filename

        for obj in root.findall('object'):
            xmin = float(obj.find('bndbox').find('xmin').text)
            xmax = float(obj.find('bndbox').find('xmax').text)
            ymin = float(obj.find('bndbox').find('ymin').text)
            ymax = float(obj.find('bndbox').find('ymax').text)
            annotations["boxes"].append([xmin, ymin, xmax, ymax])
            annotations["labels"].append(CLASSES[obj.find('name').text.lower()])
        return annotations
    
"""
STARTER CODE FOR DYNAMICALLY CREATING A TENSORFLOW DATASET FROM A PYTORCH DATASET
src: ChatGPT

```
import tensorflow as tf

# Instantiate the PyTorch dataset
pytorch_dataset = XMLDatasetPyTorch(image_dir="path_to_images")

# Create TensorFlow dataset using `from_generator`
tf_dataset = tf.data.Dataset.from_generator(
    generator=lambda: pytorch_dataset,
    output_signature=(
        tf.TensorSpec(shape=pytorch_dataset[0][0].shape, dtype=tf.uint8),  # image
        {
            "boxes": tf.TensorSpec(shape=[None, 4], dtype=tf.float32),      # boxes
            "labels": tf.TensorSpec(shape=[None], dtype=tf.int64),          # labels
            "image_id": tf.TensorSpec(shape=(), dtype=tf.int32),            # image_id
            "area": tf.TensorSpec(shape=[None], dtype=tf.float32),          # area
            "iscrowd": tf.TensorSpec(shape=[None], dtype=tf.int64)          # iscrowd - read note above on what iscrowd is
        }
    )
)

# Iterate over the TensorFlow dataset
for image, target in tf_dataset:
    # Process each sample in the TensorFlow dataset
    pass
```
"""
