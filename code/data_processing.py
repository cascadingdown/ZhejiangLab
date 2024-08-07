#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder

# Define artificial and natural classes
artificial_classes = ['buildings', 'denseresidential',  'airplane', 'storagetanks',
                      'mediumresidential', 'freeway', 'parkinglot', 'intersection',
                      'mobilehomepark', 'overpass', 'runway', 'tenniscourt', 'harbor', 'sparseresidential']
natural_classes = ['agricultural', 'baseballdiamond', 'beach', 
                   'chaparral', 'forest', 'golfcourse', 'river']

data_dir = 'UCMerced_LandUse/Images/'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImageFolder(root=data_dir, transform=transform)

# Update labels
for idx in range(len(dataset)):
    class_name = dataset.classes[dataset.targets[idx]]
    if class_name in artificial_classes:
        dataset.targets[idx] = 0
    elif class_name in natural_classes:
        dataset.targets[idx] = 1

# Split dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

class CustomDataset(Dataset):
    def __init__(self, orig_dataset, indices):
        self.orig_dataset = orig_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.orig_dataset[self.indices[idx]]
        new_label = 0 if self.orig_dataset.classes[label] in artificial_classes else 1
        return img, new_label

# Create train and test datasets
train_dataset = CustomDataset(dataset, train_dataset.indices)
test_dataset = CustomDataset(dataset, test_dataset.indices)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

