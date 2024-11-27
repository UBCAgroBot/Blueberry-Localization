from scripts.trainers.faster_rcnn import train_and_evaluate_frcnn, data_integrity_check
from build_confusion_matrix import evaluate_frcnn

if __name__ == '__main__':
  train_and_evaluate_frcnn()
  # evaluate_frcnn()
  # data_integrity_check()
