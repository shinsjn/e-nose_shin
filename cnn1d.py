# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:56:01 2021

@author: Jb
"""
import torch
import torch.nn.functional as F
class CNN_1d(torch.nn.Module):
    
    def __init__(self, channel1, channel2, lin1, out_size, ker_size, pool_size):
        super().__init__()
        self.stride = 1
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=channel1, kernel_size=ker_size, stride=self.stride)
        self.bn1 = torch.nn.BatchNorm1d(channel1)
        self.pool = torch.nn.AvgPool1d(kernel_size=pool_size)
        self.conv2 = torch.nn.Conv1d(in_channels=channel1, out_channels=channel2, kernel_size=ker_size, stride=self.stride)
        self.bn2 = torch.nn.BatchNorm1d(channel2)
        self.globalMaxPool = torch.nn.AdaptiveMaxPool1d(channel2)
        self.fc1 = torch.nn.Linear(channel2*channel2, lin1)
        self.fc2 = torch.nn.Linear(lin1, out_size)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        # x = self.bn2(x)
        x = self.pool(x)
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        x = self.globalMaxPool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
