import torch
import torch.nn as nn
import tqdm

from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import get_loaders, test_fn, save_predictions_as_imgs

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
epochs = 5
LOAD_MODEL = False

# Train Function
def train_fn(dataloader, model, opt, loss_fn):
    epochs = 5
    total_step = len(dataloader)
    for i, (x, y) in enumerate(dataloader):
        # Move tensors to the configured device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(x)
        y = y.unsqeeze(1)
        loss = loss_fn(outputs, y)

        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

def main():

    # Model
    model = UNET(in_channels=3, out_channels=1).to(device)

    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dl, test_dl = get_loaders(
                            train_dir = "./Dataset/Carvana/train",
                            train_maskdir =  "./Dataset/Carvana/train_masks",
                            test_dir = "./Dataset/Carvana/test",
                            test_maskdir =  "./Dataset/Carvana/test_masks")

    #Load checkpoint
    if LOAD_MODEL:
        model.load_state_dict(torch.load("model.ckpt")["state_dict"])

    for epoch in range(epochs):
        train_fn(train_dl, model, opt, loss_fn)

    # save checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    # check accuracy
    test_fn(test_dl, model)

    # print some examples to a folder
    save_predictions_as_imgs(test_dl, model, folder="saved_images/")


if __name__ == "__main__":
    main()