import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

classes = []
containing_half_name = []
containing_half_annotations = []

'object', 'half', 'json'

def parse_xml(xml_file_path):
    image_path = xml_file_path[:-3] + 'jpg'
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    annotations = { "boxes": [], "labels": [] }
    filename = root.find('filename').text
    annotations["filename"] = filename

    contains_half = False

    for obj in root.findall('object'):
        xmin = float(obj.find('bndbox').find('xmin').text)
        xmax = float(obj.find('bndbox').find('xmax').text)
        ymin = float(obj.find('bndbox').find('ymin').text)
        ymax = float(obj.find('bndbox').find('ymax').text)
        label = obj.find('name').text.lower()
        if label == 'half':
            annotations["boxes"].append([xmin, ymin, xmax, ymax])
        if label == 'half':
            annotations["labels"].append(label)
        if label not in classes:
            classes.append(label)
        if label == 'half':
            contains_half = True
            containing_half_name.append(image_path)
    if contains_half:
      containing_half_annotations.append(annotations)
    return annotations

images_and_xml = list(sorted(os.listdir('../../data/blueberries/all_data')))

for idx in tqdm(range(0, len(images_and_xml), 1)):
   xml_path = '../../data/blueberries/all_data' + '/' + images_and_xml[idx]
   if xml_path.endswith('.xml'):
      parse_xml(xml_path)

print("CLASSES")
print(classes)

print("CONTAINS HALF")
# print(containing_half_name)
print(len(containing_half_name) / len(images_and_xml) / 2)

# visualize bounding boxes over contains_half
import cv2
import numpy as np

def overlay_boxes(image, boxes, labels):
    overlay = image.copy()

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (139, 0, 0)  # Green color for bounding box
        thickness = 2  # Thickness of bounding box
        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # Write label
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Combine the original image with the overlay
    result = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    return result

if __name__ == "__main__":
    for image, annotation in tqdm(zip(containing_half_name[:10], containing_half_annotations[:10])):
        image = cv2.imread(image)
        boxes = annotation['boxes'] 
        labels = annotation['labels']

        # Overlay bounding boxes on the image
        result_image = overlay_boxes(image, boxes, labels)

        # Display result
        cv2.imshow("Result", result_image)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()