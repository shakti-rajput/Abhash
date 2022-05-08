import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3): 
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=48 * 48 * 4, out_features=5)

    def forward(self, x):
        # TODO: compute forward pass
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        # Remember to flatten the feature map using x.view
        # must have dimentions: N,
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x

