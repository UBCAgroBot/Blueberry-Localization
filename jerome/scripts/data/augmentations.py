from torchvision.transforms import v2 as T
import torch

def get_transform(train):
    transforms = []
    if train:
        # 0.5 is probability of an arbitrary image being flipped
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
