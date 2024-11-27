from tqdm import tqdm
from scripts.data.augmentations import get_transform
from scripts.data.xml_dataset import XMLDatasetPyTorch
from scripts.models import frcnn
from scripts.helpers import utils
from scripts.config.config import TEST_DIR, NUM_CORES, XMIN_IDX, YMIN_IDX, XMAX_IDX, YMAX_IDX, CLASS_CODES_MAP
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

ACCURACY_THRESHOLD = 0.5
MODEL_NAME = 'saved_models/model_frcnn_32.pth'
device = torch.device('cpu')
num_classes = 3

"""
Calculate Intersection over Union (IoU) between two bounding boxes.

Args:
gt_bbox (dict): Ground truth bounding box with keys 'xmin', 'xmax', 'ymin', 'ymax'.
pred_bbox (dict): Prediction bounding box with keys 'xmin', 'xmax', 'ymin', 'ymax'.

Returns:
float: Intersection over Union (IoU) value.
"""
def calculate_iou(gt_bbox, pred_bbox):
    # Calculate the coordinates of the intersection rectangle
    x_left = max(gt_bbox[XMIN_IDX], pred_bbox[XMIN_IDX])
    y_top = max(gt_bbox[YMIN_IDX], pred_bbox[YMIN_IDX])
    x_right = min(gt_bbox[XMAX_IDX], pred_bbox[XMAX_IDX])
    y_bottom = min(gt_bbox[YMAX_IDX], pred_bbox[YMAX_IDX])

    # Check for no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of each bounding box
    gt_bbox_area = (gt_bbox[XMAX_IDX] - gt_bbox[XMIN_IDX]) * (gt_bbox[YMAX_IDX] - gt_bbox[YMIN_IDX])
    pred_bbox_area = (pred_bbox[XMAX_IDX] - pred_bbox[XMIN_IDX]) * (pred_bbox[YMAX_IDX] - pred_bbox[YMIN_IDX])

    # Calculate the union area
    union_area = gt_bbox_area + pred_bbox_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_iou_matrix(gt_coordinates, predicted_coordinates, gt_labels, predicted_labels):
  # Compute IoU between predicted and ground truth bounding boxes
  iou_matrix = np.zeros((len(gt_labels), len(predicted_labels))) # WOAH! The length of the gt_label and predicted_labels
                                                                 # doesn't have to be the same!!!
  # Number of rows of IOU matrix === number of ground truth bounding boxes                                                         
  # since we want to match every ground truth bounding box to the best prediction boxes
  # Note: This means that our metrics currently don't penalize our model for making very wild 
  #       predictions. If my models makes 50 predictions when there are 10 ground truth bounding 
  #       boxes, then I will not penalize the 40~ predictions that may be completely off the mark
  # QUESTION: For a robot, if we only care about picking EVERY blueberry, then this might 
  # not matter so much, but is there a convenient metric that we can keep track of this in?
  for i, gt_bbox in enumerate(gt_coordinates):
      for j, pred_bbox in enumerate(predicted_coordinates):
          iou = calculate_iou(gt_bbox, pred_bbox)  # Implement calculate_iou function
          iou_matrix[i, j] = iou
  return iou_matrix

def match_ground_truth_labels_to_predicted_labels(gt_labels, predicted_labels, iou_matrix):
  # Use Hungarian algorithm to find the best matches
  # Assigns every ground truth label to a predicted label (indices list will be same length)
  gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

  # Assign predicted labels to ground truth labels based on matching
  matched_gt_labels = [gt_labels[i] for i in gt_indices]
  matched_pred_labels = [predicted_labels[j] for j in pred_indices]
  return (matched_gt_labels, matched_pred_labels)

def get_data_loader_test():
  dataset_test = XMLDatasetPyTorch(TEST_DIR, get_transform(train=False))

  # keep batch size at 1 since we want to apply IOU matching Hungarian algorithm to only 
  # one image at a time
  data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_CORES, 
    collate_fn=utils.collate_fn
  )

  return data_loader_test

def evaluate_frcnn():
  model = frcnn.get_model(num_classes)
  model.to(device)
  model.load_state_dict(torch.load(MODEL_NAME, map_location=torch.device('cpu')))
  model.eval()

  RANDOM_SEED = 42
  torch.manual_seed(RANDOM_SEED)

  data_loader_test = get_data_loader_test()

  all_matched_gt_labels = []
  all_matched_pred_labels = []

  # Iterate over test data to build a confusion matrix
  for image, target in tqdm(data_loader_test):
    gt_labels, gt_coordinates = [x.item() for x in target[0]['labels']], target[0]['boxes']

    # Initialize lists to store predicted labels and coordinates
    predicted_labels = []
    predicted_coordinates = []

    prediction = model(image)[0]

    for idx, label in enumerate(prediction['labels']):
      predicted_labels.append(label.item())
      bbox = prediction['boxes'][idx].tolist()
      predicted_coordinates.append(bbox)

    iou_matrix = calculate_iou_matrix(gt_coordinates, predicted_coordinates, gt_labels, predicted_labels)
    matched_gt_labels, matched_pred_labels = match_ground_truth_labels_to_predicted_labels(gt_labels, predicted_labels, iou_matrix)
    all_matched_gt_labels += matched_gt_labels
    all_matched_pred_labels += matched_pred_labels

  print(len(all_matched_gt_labels))
  print(all_matched_gt_labels)
  print(len(all_matched_pred_labels))

  # Compute confusion matrix
  conf_matrix = confusion_matrix(all_matched_gt_labels, all_matched_pred_labels, labels=[1, 2])
  print(conf_matrix)

  # Visualize confusion matrix
  display = ConfusionMatrixDisplay(conf_matrix, display_labels=[CLASS_CODES_MAP[1], CLASS_CODES_MAP[2]])
  display.plot()
  plt.savefig('./confusion_matrix.png')
  plt.show()

  # Optional code to run that calculates average precision and recall
  # evaluate(model, data_loader_test, device=device)
