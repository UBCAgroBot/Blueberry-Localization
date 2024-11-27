import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # cool trick, ensure 1:1 correspondence between images and 
        # corresponding labels by naming them in a 1:1 manner and sorting them`
        # when you load them into application state
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))[:10] # TODO - remove
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))[:10]

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # removes all duplicates instance segementation masks 
        # before returning them as a tuple
        obj_ids = torch.unique(mask)
        # removes background mask ID
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # HOW DOES THIS CREATE BINARY MASKS?
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # How does the mask look?
        boxes = masks_to_boxes(masks)

        # comma after num_objs in original code is purely stylistic
        labels = torch.ones((num_objs), dtype=torch.int64)

        image_id = idx

        # What does a single box look like?
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd (? - WHY DOES MAKING THIS 
        # SUPPOSITION MATTER?)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        img = tv_tensors.Image(img)

        target = {}
        # What is 'XYXY' format?
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY",
                                                   canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        # Is 'crowd' a separate class?
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)






