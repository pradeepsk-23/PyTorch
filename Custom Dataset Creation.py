import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd

from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DriveandAct(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

# Load Data
dataset = DriveandAct(
    csv_file="midlevel.chunks_90.csv",
    root_dir="../Dataset/Annotations/activities_3s/inner_mirror",
    transform=transforms.ToTensor())