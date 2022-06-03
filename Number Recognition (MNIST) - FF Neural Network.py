import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset (images and labels)
train_dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

# DataLoader (input pipeline)
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle = True)
test_dl = DataLoader(test_dataset, batch_size)

# Fully connected neural network with one hidden layer
input_size = 28*28
hidden_size = 392
output_size = 10 # number of classes

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
# F.cross_entropy computes softmax internally
loss_fn = F.cross_entropy
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
epochs = 5
total_step = len(train_dl)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dl):
        # Reshape images to (batch_size, input_size)  
        # Move tensors to the configured device
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (i+1) % 600 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dl:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'nr_ffnn.ckpt')