import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from os.path import exists
import matplotlib.pyplot as plt
from collections import Counter
import pprint

IMAGE_DIR = 'data/blueberries/all_data'
TRAIN_DIR = 'data/blueberries/train'
TEST_DIR = 'data/blueberries/test'
VALIDATE_DIR = 'data/blueberries/validate'

from scripts.config.config import CLASSES, CLASS_CODES_MAP

CLASSES = {
  "background": 0,
  "green": 1, 
  "unripe": 1, 
  "half": 1,   # half-ripe
  "immature": 1,
  "ripe": 2,
}

CLASS_CODES_MAP = [
  'background', # 0 => background
  'unripe',     # 1 => unripe
  'ripe',       # 2 => ripe
]

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def get_metrics():
    images_and_xml = list(sorted(listdir_nohidden(IMAGE_DIR)))
    counter_dict = {
      'train_ripe_counter': 0,
      'train_unripe_counter': 0,
      'test_ripe_counter': 0,
      'test_unripe_counter': 0,
      'validate_ripe_counter': 0,
      'validate_unripe_counter': 0
    }

    for dir in tqdm([TRAIN_DIR, TEST_DIR, VALIDATE_DIR]):
      images_and_xml = list(sorted(listdir_nohidden(dir)))
      for idx in tqdm(range(0, len(images_and_xml))):
        # ignore images
        if not images_and_xml[idx].endswith('.xml'):
          continue

        xml_path = f"{dir}/{images_and_xml[idx]}"
        [ripe_count, unripe_count] = get_class_counts(xml_path)
        update_count(dir, counter_dict, ripe_count, unripe_count)
    
    plot_metrics(counter_dict)

def plot_metrics(counter_dict):
  plt.figure(figsize=(10, 6))
  bar_width = 0.35
  directories = ['train', 'test', 'validate']
  index = range(len(directories))

  ripe_counts = [counter_dict['train_ripe_counter'], counter_dict['test_ripe_counter'], counter_dict['validate_ripe_counter']]
  unripe_counts = [counter_dict['train_unripe_counter'], counter_dict['test_unripe_counter'], counter_dict['validate_unripe_counter']]

  plt.barh(index, ripe_counts, bar_width, label='Ripe')
  plt.barh(index, unripe_counts, bar_width, left=ripe_counts, label='Unripe')

  plt.xlabel('Number of XML files')
  plt.ylabel('Directories')
  plt.title('Number of XML files containing "ripe" and "unripe" classes in each directory')
  plt.yticks(index, directories)
  plt.legend()

  plt.tight_layout()
  plt.show()

def update_count(dir, counter_dict, ripe_count, unripe_count):
  if dir == TRAIN_DIR:
    counter_dict['train_ripe_counter'] += ripe_count
    counter_dict['train_unripe_counter'] += unripe_count
  elif dir == TEST_DIR:
    counter_dict['test_ripe_counter'] += ripe_count
    counter_dict['test_unripe_counter'] += unripe_count
  else:
    counter_dict['validate_ripe_counter'] += ripe_count
    counter_dict['validate_unripe_counter'] += unripe_count

def get_class_counts(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    ripe_count = 0
    unripe_count = 0

    for obj in root.findall('object'):
        label = obj.find('name').text.lower()
        class_code = CLASSES[label]               # number
        final_label = CLASS_CODES_MAP[class_code] # one of 'background', 'ripe', 'unripe'
        if final_label == 'ripe':
          ripe_count += 1
        elif final_label == 'unripe':
          unripe_count += 1
        else: 
          print(f"FINAL LABEL OTHER THAN RIPE OR UNRIPE: {final_label}")
    return [ripe_count, unripe_count]

get_metrics()
