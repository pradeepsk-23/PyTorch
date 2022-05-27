import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader

#MNIST DATASET
dataset = MNIST(root='data/', download=True)

#IMAGE TO TENSOR TRANSFORM
dataset = MNIST(root = 'data/', train=True, transform = transforms.ToTensor())

#TRAINING AND VALIDATION DATASET SPLIT
train_ds, val_ds = random_split(dataset, [50000, 10000])

#DATALOADER
batch_size = 100
train_dl = DataLoader(train_ds, batch_size, shuffle = True)
val_dl = DataLoader(val_ds, batch_size, shuffle=True)

#MODEL
input_size = 28*28
num_classes = 10
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = MnistModel()

# TRAINING THE MODEL

def fit(epochs, lr, model, train_dl, val_dl):
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    history = []                                        # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        # Training Phase 
        for xb, yb in train_dl:
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
         
        # Validation phase
        result = evaluate(model, val_dl)
        model.epoch_end(epoch, result)
         
    return history

def evaluate(model, val_dl):
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)

#ACCURACY

def accuracy(out, yb):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == yb).item() / len(preds))

print(fit(10 , 0.001, model, train_dl, val_dl))

#PSEUDO CODE
# for epoch in range(num_epochs):
    # Training phase
    # for batch in train_loader:
        # Generate predictions
        # Calculate loss
        # Compute gradients
        # Update weights
        # Reset gradients
    # Validation phase
    # for batch in val_loader:
        # Generate predictions
        # Calculate loss
        # Calculate metrics (accuracy etc.)
    # Calculate average validation loss & metrics
    
    # Log epoch, loss & metrics for inspection

# TEST DATASET

# test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

# # TEST PHASE

# def predict_image(img, model):
#     xb = img.unsqueeze(0)
#     yb = model(xb)
#     _, preds = torch.max(yb, dim=1)
#     return preds[0].item()


# img, label = test_dataset[0]
# plt.imshow(img[0], cmap='gray')
# print('Label:', label, ', Predicted:', predict_image(img, model))

