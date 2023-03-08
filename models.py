import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from tools import *


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 18, 300),
            nn.BatchNorm1d(300),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(300, 100),
            nn.BatchNorm1d(100),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(50, 20),
            nn.BatchNorm1d(20),
            nn.ELU(inplace=True),
            nn.Linear(20, 2),
            nn.Sigmoid()
        )
        self.tn = TimeNorm(1280)

    def forward(self, x):
        x = self.tn(x)
        return self.model(x)


# cnn + 双向gru
class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (2, 2), bias=False),  # 1279 * 17
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 1278 * 16
            nn.MaxPool2d((3, 2)),  # 426 * 8
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 425 * 7
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 424 * 6
            nn.MaxPool2d((4, 2)),  # 106 * 3
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (3, 2), bias=False),  # 104 * 2
            nn.MaxPool2d((4, 2)),  # 26 * 1
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(32 * 26, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True)
        )
        self.gru = nn.GRU(input_size=18, hidden_size=32, bidirectional=True)
        self.mix = nn.Linear(64 * 2, 2)
        self.tn = TimeNorm(1280)

    def forward(self, x):
        x = self.tn(x)

        x_cnn = self.cnn(x)
        x = x.reshape(1280, -1, 18)
        h_0 = torch.randn(2, x.shape[1], 32).cuda()
        x_gru, _ = self.gru(x, h_0)
        x_gru = x_gru[-1]
        x = torch.cat([x_cnn, x_gru], dim=1)
        x = self.mix(x)
        return torch.sigmoid(x)


# canet+gruta
class CANet_GRUTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (2, 2), bias=False),  # 1279 * 17
            nn.BatchNorm2d(32),
            CAlayer(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 1278 * 16
            nn.MaxPool2d((3, 2)),  # 426 * 8
            nn.BatchNorm2d(32),
            CAlayer(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 425 * 7
            nn.BatchNorm2d(32),
            CAlayer(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 424 * 6
            nn.MaxPool2d((4, 2)),  # 106 * 3
            nn.BatchNorm2d(32),
            CAlayer(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (3, 2), bias=False),  # 104 * 2
            nn.MaxPool2d((4, 2)),  # 26 * 1
            nn.BatchNorm2d(32),
            CAlayer(32),
            nn.ELU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(32 * 26, 256),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True)
        )
        self.gru = nn.GRU(input_size=18, hidden_size=32, bidirectional=True, num_layers=1)
        self.linear = nn.Linear(32 * 2, 2)
        self.tn = TimeNorm(1280)
        self.talayer = TAlayer(hidden_size=32, bidirectional=True)
        self.mix = nn.Linear(64 * 2, 2)

    def forward(self, x):
        x = self.tn(x)

        x_cnn = self.cnn(x)
        x = x.reshape(1280, -1, 18)
        h_0 = torch.randn(2 * 1, x.shape[1], 32).cuda()
        x_gru, h_n = self.gru(x, h_0)
        x_gru = self.talayer(x_gru)

        x = torch.cat([x_cnn, x_gru], dim=1)
        x = self.mix(x)
        return torch.sigmoid(x)


# cnn - et
class CNN__ET(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 32
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 4, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=6, norm=nn.LayerNorm(d_model))
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (2, 2), bias=False),  # 1279 * 17
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 1278 * 16
            nn.MaxPool2d((2, 2)),  # 639 * 8
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (3, 2), bias=False),  # 637 * 7
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 636 * 6
            nn.MaxPool2d((2, 2)),  # 318 * 3
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (3, 2), bias=False),  # 316 * 2
            nn.MaxPool2d((2, 2)),  # 158 * 1
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.tn = TimeNorm(1280)
        self.linear = nn.Linear(32, 2)

    def forward(self, x):
        x = self.tn(x)

        x = self.cnn(x)
        x = x.squeeze(-1).permute(0, 2, 1)

        x = self.encoder(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=1)
        x = self.linear(x)
        return torch.sigmoid(x)


# cnn - gru
class CNN__GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (2, 2), bias=False),  # 1279 * 17
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 1278 * 16
            nn.MaxPool2d((2, 2)),  # 639 * 8
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (3, 2), bias=False),  # 637 * 7
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 636 * 6
            nn.MaxPool2d((2, 2)),  # 318 * 3
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (3, 2), bias=False),  # 316 * 2
            nn.MaxPool2d((2, 2)),  # 158 * 1
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )
        self.gru = nn.GRU(input_size=32, hidden_size=32, bidirectional=True, num_layers=1)
        self.tn = TimeNorm(1280)
        self.linear = nn.Linear(64, 2)

    def forward(self, x):
        x = self.tn(x)

        x = self.cnn(x)
        x = x.squeeze(-1).permute(2, 0, 1)
        h_0 = torch.randn(2 * 1, x.shape[1], 32).cuda()
        x, _ = self.gru(x, h_0)
        x = x[-1]

        x = self.linear(x)
        return torch.sigmoid(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (2, 2), bias=False),  # 1279 * 17
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 1278 * 16
            nn.MaxPool2d((2, 2)),  # 639 * 8
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (3, 2), bias=False),  # 637 * 7
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (2, 2), bias=False),  # 636 * 6
            nn.MaxPool2d((2, 2)),  # 318 * 3
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, (3, 2), bias=False),  # 316 * 2
            nn.MaxPool2d((2, 2)),  # 158 * 1
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.tn = TimeNorm(1280)
        self.linear = nn.Linear(32*158, 2)

    def forward(self, x):
        x = self.tn(x)

        x = self.cnn(x)
        x = self.flatten(x)
        x = self.linear(x)
        return torch.sigmoid(x)
