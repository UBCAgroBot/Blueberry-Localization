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

TRAIN_DIR = 'data/blueberries/train'
TEST_DIR = 'data/blueberries/test'
VALIDATE_DIR = 'data/blueberries/validate'

# HYPERPARAMETERS
NUM_EPOCHS = 10

# CLOUD COMPUTING SETTINGS
NUM_CORES = 0

# BND BOX PROTOCOLS
XMIN_IDX = 0
YMIN_IDX = 1
XMAX_IDX = 2
YMAX_IDX = 3

# WANDB SPECIFIC CONSTANT 
PROJECT_NAME = "blueberry_localization_fRCNN_overfitting"