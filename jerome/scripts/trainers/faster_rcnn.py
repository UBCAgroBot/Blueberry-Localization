from scripts.helpers.engine import train_one_epoch, evaluate
from scripts.data.augmentations import get_transform
from scripts.data.xml_dataset import XMLDatasetPyTorch
from scripts.models.frcnn import get_model
from scripts.helpers import utils
from scripts.config.config import TRAIN_DIR, TEST_DIR, VALIDATE_DIR,\
        NUM_CORES, NUM_EPOCHS, PROJECT_NAME
import scripts.logging.wandb as wandb
import torch
from tqdm import tqdm

def train_and_evaluate_frcnn():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)

    num_classes = 3
    dataset = XMLDatasetPyTorch(TRAIN_DIR, get_transform(train=True))
    dataset_test = XMLDatasetPyTorch(TEST_DIR, get_transform(train=False))

    data_loader_train = torch.utils.data.DataLoader(
      dataset,
      batch_size=2,
      shuffle=True,
      num_workers=NUM_CORES,
      collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
      dataset_test,
      batch_size=1,
      shuffle=False,
      num_workers=NUM_CORES, 
      collate_fn=utils.collate_fn
    )

    model = get_model(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
      params,
      lr=0.005,
      betas=(0.9, 0.99),
      weight_decay=0 # e.g., the lambda term used in L1 and L2 regularization
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1 # every 3 epochs, multiple currently learning rate by 0.1
    )

    wandb.initialize_wandb()
    wandb.create_run(PROJECT_NAME)

    print(f'--- TRAINING ALL EPOCHS ---')

    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f"--- TRAINING EPOCH {epoch} ---")
        loss_training = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        wandb.log_train_loss(loss_training)
        wandb.increment_step()
        lr_scheduler.step()
        # Comment out 2 lines below for faster training
        print(f'--- EVAULATING MODEL FOR EPOCH {epoch} ----')
        evaluate(model, data_loader_test, device=device)
        if epoch % 2 == 0:
          # Increase interval between saves for faster training
          print(f'--- SAVING THE MODEL ---')
          torch.save(model.state_dict(), f'./saved_models/model_frcnn_{epoch}.pth')

def data_integrity_check():
    dataset = XMLDatasetPyTorch(TRAIN_DIR, get_transform(train=True))
    dataset_test = XMLDatasetPyTorch(TEST_DIR, get_transform(train=False))
    print(len(dataset))
    print(len(dataset_test))

    train_c = 0
    for (image, target) in dataset:
      train_c += 1

    test_c = 0
    for (image, target) in dataset_test:
      print("success train!")
      test_c += 1
    
    print(train_c)
    print(test_c)
        

