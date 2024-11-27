import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from os.path import exists
import shutil
import matplotlib.pyplot as plt
import cv2
from collections import Counter

IMAGE_DIR = '../../data/blueberries/all_data'
TRAIN_DIR = '../../data/blueberries/train'
TEST_DIR = '../../data/blueberries/test'
VALIDATE_DIR = '../../data/blueberries/validate'

train_ripe_counter = 0
train_unripe_counter = 0

test_ripe_counter = 0
test_unripe_counter = 0

validate_ripe_counter = 0
validate_unripe_counter = 0

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def get_images_dimensions_averages():
    images_and_xml = list(sorted(listdir_nohidden(IMAGE_DIR)))

    data = []
    widths = []
    heights = []

    for dir in tqdm([TRAIN_DIR, TEST_DIR, VALIDATE_DIR]):
      images_and_xml = list(sorted(listdir_nohidden(dir)))
      for idx in tqdm(range(0, len(images_and_xml), 2)):
        jpg_path = dir + '/' + images_and_xml[idx]
        image = cv2.imread(jpg_path)

        if image is None:
          continue

        # cv2.imread returns an empty matrix if it is unable to read the image
        if len(image) == 0:
          continue

        height = image.shape[0]
        width = image.shape[1]
        area = height * width 

        widths.append(width)
        heights.append(height)
        data.append(area)
    
    print(f"Average width across all directories: {sum(widths) / len(widths)}")
    print(f"Average height across all directories: {sum(heights) / len(heights)}")
    print(f"Average area across all directories: {sum(data) / len(data)}")

get_images_dimensions_averages()

