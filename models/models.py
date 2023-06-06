import torch
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Module

class UndergroundModel(Module):
    def __init__(self):
        super(UndergroundModel, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=10, stride=1, padding=2)
        self.conv2 = Conv2d(16, 32, kernel_size=10, stride=1, padding=1)
        self.conv3 = Conv2d(32, 64, kernel_size=7, stride=1, padding=1)
        self.fc1 = Linear(7744, 16)
        self.fc2 = Linear(16, 16)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
