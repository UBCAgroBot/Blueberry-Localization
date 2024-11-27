from scripts.helpers.engine import train_one_epoch, evaluate
from scripts.data.augmentations import get_transform
from scripts.data.dataset import PennFudanDataset
from scripts.models.mask_rcnn import get_model_instance_segementation
from scripts.helpers import utils
import torch
from tqdm import tqdm

# Q: what is the difference between fork and spawn in relation to PyTorch?
#    What is __name__?
# Q: Python spawn multithreading?
def train_and_evaluate_maskrcnn():
    # MPS issues
    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    num_classes = 2
    dataset = PennFudanDataset('data/pedestrians', get_transform(train=True))
    # setting train=False ensures we don't apply a random horizontal flip to our images
    dataset_test = PennFudanDataset('data/pedestrians', get_transform(train=False))

    data_loader_train = torch.utils.data.DataLoader(
      dataset,
      batch_size=2,
      shuffle=True,
      num_workers=0,
      collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
      dataset_test,
      batch_size=1,
      shuffle=False,
      num_workers=0, 
      collate_fn=utils.collate_fn
    )

    model = get_model_instance_segementation(num_classes)

    model.to(device)

    # What parameters of a RCNN is does not require a gradient?
    # Do these parameters remain constant?
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
      params,
      lr=0.005,
      momentum=0.9,       # beta
      weight_decay=0.0005 # e.g., the lambda term used in L1 and L2 regularization
    )

    # Q: What would a negative LR look like? What would a complex LR look like?
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1 # every 3 epochs, multiple currently learning rate by 0.1
    )

    num_epochs = 1

    print(f'--- TRAINING ALL EPOCHS ---')
    for epoch in tqdm(range(num_epochs)):
        print(f"--- TRAINING EPOCH {epoch} ---")
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        print(f'--- EVAULATING MODEL FOR EPOCH {epoch} ----')
        evaluate(model, data_loader_test, device=device)

    print(f'--- SAVING THE MODEL ---')
    torch.save(model.state_dict(), './saved_models/model_mask_rcnn.pth')


