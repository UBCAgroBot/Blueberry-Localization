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

def get_image_size_histogram():
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

    fig = plt.figure(figsize=(20, 6))

    counter = Counter(data)
    x = list(counter.keys())
    y = list(counter.values())
    width = (max(x) - min(x)) / len(x)

    ax1 = fig.add_subplot(3, 1, 1) 
    ax1.bar(x, y, width)
    # since data range is humongous
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Image Area')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Image Size Frequencies')

    """
    xticks = [10e1, 10e2, 10e3]
    xticklabels = ['10¹', '10²', '10³']
    yticks = [10e1, 10e2, 10e3]
    yticklabels = ['10¹', '10²', '10³']
    ax2.set_xticks(xticks, xticklabels)
    ax2.set_yticks(yticks, yticklabels)
    """
    ax2 = fig.add_subplot(3, 1, 2) 

    counter = Counter(widths)
    x1 = list(counter.keys())
    y1 = list(counter.values())
    width = (max(x1) - min(x1)) / len(x1)
    ax2.bar(x1, y1, width)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    xticks = [10e1, 10e2, 10e3]
    xticklabels = ['10¹', '10²', '10³']
    ax2.set_xticks(xticks, xticklabels)
    ax2.set_xlabel('Side Dimension')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Side Dimension Magnitude vs Frequency')

    ax3 = fig.add_subplot(3, 1, 3) 
    counts = [ sum(1 for width, height in zip(widths, heights) if width == height),
               sum(1 for width, height in zip(widths, heights) if width != height) ]

    # THIS LOGIC MIGHT BE WRONG!!!
    print(counts[0])
    print(counts[1])
    categories = ["Match", "No Match"]
    ax3.bar(categories, counts)
    ax3.set_xlabel('Match or No Match')
    ax3.set_ylabel('Number')
    ax3.set_title('Does width == height?')


    plt.tight_layout()
    plt.show()

def restore_data():
    for dir in [TRAIN_DIR, TEST_DIR, VALIDATE_DIR]:
      images_and_xml = list(sorted(listdir_nohidden(dir)))
      for idx in range(0, len(images_and_xml), 2):
        jpg_path = dir + '/' + images_and_xml[idx]
        xml_path = dir + '/' + images_and_xml[idx][:-3] + 'xml'

        new_jpg_path = IMAGE_DIR + '/' + images_and_xml[idx]
        new_xml_path = IMAGE_DIR + '/' + images_and_xml[idx][:-3] + 'xml'

        try: 
          shutil.move(jpg_path, new_jpg_path)
        except:
          print("do nothing")
        try:
          shutil.move(xml_path, new_xml_path)
        except: 
          print("do nothing")

def split_data():
    train_idx = 0
    test_idx = 0
    validate_idx = 0

    train_c = 0
    test_c = 0
    validate_c = 0

    images_and_xml = list(sorted(listdir_nohidden(IMAGE_DIR)))

    for idx in range(0, len(images_and_xml), 2):
        jpg_path = IMAGE_DIR + '/' + images_and_xml[idx]
        xml_path = IMAGE_DIR + '/' + images_and_xml[idx][:-3] + 'xml'

        if train_idx == TRAIN_COUNTER and test_idx == TEST_COUNTER and validate_idx == VALIDATE_COUNTER:
            train_idx = 0
            test_idx = 0
            validate_idx = 0

        new_jpg_path = ''
        new_xml_path = ''
        
        if train_idx != TRAIN_COUNTER:
            new_jpg_path = TRAIN_DIR + '/' + images_and_xml[idx]
            new_xml_path = TRAIN_DIR + '/' + images_and_xml[idx][:-3] + 'xml'
            train_idx += 1
            train_c += 2
        elif test_idx != TEST_COUNTER:
            new_jpg_path = TEST_DIR + '/' + images_and_xml[idx]
            new_xml_path = TEST_DIR + '/' + images_and_xml[idx][:-3] + 'xml'
            test_idx += 1
            test_c += 2
        elif validate_idx != VALIDATE_COUNTER:
            new_jpg_path = VALIDATE_DIR + '/' + images_and_xml[idx]
            new_xml_path = VALIDATE_DIR + '/' + images_and_xml[idx][:-3] + 'xml'
            validate_idx += 1
            validate_c += 2

        try: 
          shutil.move(jpg_path, new_jpg_path)
        except:
          print("do nothing")
        try:
          shutil.move(xml_path, new_xml_path)
        except: 
          print("do nothing")

    print(f"TRAIN_COUNTER {train_c}")
    print(f"TEST_COUNTER {test_c}")
    print(f"VALIDATE_COUNTER {validate_c}")

def remove_singles():
  images_and_xml = list(sorted(listdir_nohidden(IMAGE_DIR)))
  for file in images_and_xml:
    jpg_path = IMAGE_DIR + '/' + file
    xml_path = IMAGE_DIR + '/' + file[:-3] + 'xml'
    if not exists(jpg_path) or not exists(xml_path):
        try:
          os.remove(jpg_path)
          print("REMOVING SINGLE")
        except:
          print("a")

        try: 
          os.remove(xml_path)
          print("REMOVING SINGLE")
        except:
          print("a")

get_image_size_histogram()

"""
def parse_xml(xml_file_path):
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
        label = obj.find('name').text.lower()
        annotations["boxes"].append([xmin, ymin, xmax, ymax])
        annotations["labels"].append(label)
    return annotations

images_and_xml = list(sorted(os.listdir('../../data/blueberries/all_data')))

# remove every image that doesn't have a corresponding XML file
# remove every XML file that doesn't have a corresponding image
image_counter = 0
xml_counter = 0
for file in tqdm(images_and_xml):
   file_path = '../../data/blueberries/all_data' + '/' + file

   xml_path = file_path[:-3] + 'xml'
   jpg_path = file_path[:-3] + 'jpg'


   try: 
     annotations = parse_xml(xml_path) 
   except FileNotFoundError: 
     try:
       os.remove(jpg_path)
     except FileNotFoundError:
       print('Not found')
     continue

   if not exists(xml_path):
      try:
        os.remove(jpg_path)
        image_counter += 1
      except FileNotFoundError:
        print('Not found')
   elif not exists(jpg_path):
      try:
        os.remove(xml_path)
        xml_counter += 1
      except FileNotFoundError:
        print('Not found')

   for box in annotations["boxes"]:
     if box[3] == box[1] or box[2] == box[0]:
      try:
        os.remove(xml_path)
        xml_counter += 1
      except FileNotFoundError:
        print('Not found')
        # do nothing

      try:
        os.remove(jpg_path)
        image_counter += 1
      except FileNotFoundError:
        print('Not found')
        # do nothing

print(f"IMAGES DELETED {image_counter}")
print(f"XML DELETED {xml_counter}")
print((image_counter + xml_counter) / len(images_and_xml))
"""
