import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10 dataset (images and labels)
train_dataset = CIFAR10(root='Data/CIFAR10', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = CIFAR10(root='Data/CIFAR10', train=False, transform=transforms.ToTensor(), download=True)

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size)

# Fully connected neural network with one hidden layer
output_size = 10

class ConvNet(nn.Module):
    def __init__(self, output_size):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            # Output pixel = {[Input Size(32 in this case) - K(Kernel) + 2P(Padding)] / S(Stride)} + 1. Default Stride=1, Padding=0.
            # out = [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # out = [64, 16, 16]
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), # out = [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # out = [128, 8, 8]
        self.fc = nn.Linear(8*8*128, output_size)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Model
model = ConvNet(output_size).to(device)

# Loss and optimizer
# F.cross_entropy computes softmax internally
loss_fn = F.cross_entropy
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
epochs = 10
total_step = len(train_dl)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dl):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (i+1) % 500 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dl:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))