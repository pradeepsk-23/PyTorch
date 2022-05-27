# Creating a model that predicts crop yields for apples and oranges (target variables)
# by looking at the average temperature, rainfall, and humidity (input variables or features)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

train_ds = TensorDataset(inputs, targets)
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = nn.Linear(3,2)
for xb, yb in train_dl:
    print(xb.shape)

# Utility function to train the model
def fit(epochs, model, train_dl):
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    
    # Repeat for given number of epochs
    for epoch in range(epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            out = model(xb)
            
            # 2. Calculate loss
            loss = F.mse_loss(out, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        # if (epoch+1) % 100 == 0:
        #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

fit(1000, model, train_dl)

preds = model(inputs)

#print(preds)