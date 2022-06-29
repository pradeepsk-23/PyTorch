import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CarvanaDataset
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_loaders(train_dir, train_maskdir, test_dir, test_maskdir):

    train_transform = A.Compose([
            A.Resize(160, 240),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0),
            ToTensorV2()])
    
    test_transform = A.Compose([
            A.Resize(160, 240),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0),
            ToTensorV2()])

    train_dataset = CarvanaDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    test_dataset = CarvanaDataset(image_dir=test_dir, mask_dir=test_maskdir, transform=test_transform)

    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)

    return train_dl, test_dl

def test_fn(dataloader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        pixels = 0
        dice_score = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            predicted = torch.sigmoid(model(x))
            predicted = (predicted>0.5).float()
            correct += (predicted == y).sum()
            pixels += torch.numel(predicted)
            dice_score += (2*(predicted*y).sum()) / ((predicted+y).sum()) + 1e-8
        
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / pixels))
        print("Dice score :", dice_score/len(dataloader))
        model.train()

def save_predictions_as_imgs(dataloader, model, folder="saved_images/"):
    model.eval()
    for idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()