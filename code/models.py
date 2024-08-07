#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedMobileNetV2, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.mobilenet(x)
        return x

class ModifiedShuffleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedShuffleNet, self).__init__()
        self.shufflenet = shufflenet_v2_x1_0(pretrained=True)
        num_features = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.shufflenet(x)
        return x

class MobileShuffleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileShuffleNet, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.initial_layers = nn.Sequential(*list(mobilenet.features[:7]))
        self.middle_layers = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x

