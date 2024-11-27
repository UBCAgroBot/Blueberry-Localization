"""
CLASS_COMMENT: 
- STATEFUL module for interfacing with wanddb API

REQUIRES: 
.env file should be initialized with following key-value pair
```
export WANDB_API_KEY=your_api_key_here
```
"""
import os
from dotenv import load_dotenv
import wandb

load_dotenv()
api_key = os.getenv("WANDB_API_KEY")

""" 
WANDB STATEFUL VARIABLES
"""
run = None
current_step = None

"""
LOGGER CONSTANTS    
"""
TRAIN_LOSS = "train-loss"
VALIDATION_LOSS = "val_loss"
EPOCH = "epoch"

def initialize_wandb():
    if api_key:
        wandb.login(key=api_key)
    else:
        raise EnvironmentError("WANDB_API_KEY not found in the environment variables.")

def create_run(proj_name):
    global run, current_step
    run = wandb.init(project=proj_name)
    current_step = 0

"""
step - variable tracking current iteration of training (e.g., to the next epoch)
"""
def log_train_loss(train_loss):
    global run
    print(f"Wandb logging training loss for step {current_step}")
    run.log({ TRAIN_LOSS: train_loss }, step=current_step)

def log_val_loss(val_loss):
    global run
    print(f"Wandb logging validation loss for step {current_step}")
    run.log({ VALIDATION_LOSS: val_loss }, step=current_step)

def log_epoch(epoch):
    global run
    print(f"Wandb logging epoch step {current_step}")
    run.log({ EPOCH: epoch }, step=current_step)

def increment_step():
    global current_step
    current_step += 1

"""
PERSISTENCE
"""

"""
save a model to wandb from saved file
"""
def save_model(model_path):
    wandb.save(model_path)

"""
loads a model from cloud wandb

OUTPUT:
`filename` that can be loaded via torch.load(filename)
"""
def load_model(model_name):
    model_file = wandb.restore(model_name)
    return model_file.name

