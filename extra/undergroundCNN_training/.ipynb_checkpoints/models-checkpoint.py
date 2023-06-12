import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
import torch.nn as nn
from torch.nn import BatchNorm2d
from torch.nn import Dropout2d
from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
 
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchinfo import summary

from livelossplot import PlotLosses






class BaseModel(Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        #self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        #self.conv2 = Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        #self.fc1 = Linear(2*IMG_WIDTH*IMG_HEIGHT, 128)
        #self.fc2 = Linear(128, 10)
        #self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        #self.relu = ReLU()
        self.conv1 = Conv2d(3, 16, kernel_size=10, stride=1, padding=2)
        self.conv2 = Conv2d(16, 32, kernel_size=10, stride=1, padding=1)
        self.conv3 = Conv2d(32, 64, kernel_size=7, stride=1, padding=1)
        self.fc1 = Linear(7744, 16)
        self.fc2 = Linear(16, 16)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)
        #x = self.conv2(x)
        #x = self.relu(x)
        #x = self.maxpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc1(x)
        #x = self.relu(x)
        #x = self.fc2(x)
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


def preprocessar(imagem):
    imagem = np.array(imagem)
    data_mean = np.mean(imagem)
    data_std = np.std(imagem)
    imagem = (imagem - data_mean) / data_std
    xmax, xmin = imagem.max(), imagem.min()
    imagem = (imagem - xmin)/(xmax - xmin)
    imagem = imagem.transpose(2,1,0)
    return imagem