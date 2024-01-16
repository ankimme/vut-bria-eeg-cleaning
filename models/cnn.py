import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim


class CNN_01(nn.Module):
    def __init__(self):
        super(CNN_01, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=2, padding=0),
            nn.LeakyReLU(0.15),
            nn.Conv1d(2, 4, kernel_size=2, padding=0),
            nn.LeakyReLU(0.15),
            nn.Conv1d(4, 8, kernel_size=2, padding=0),
            nn.LeakyReLU(0.15),
            nn.Conv1d(8, 16, kernel_size=2, padding=0),
            nn.LeakyReLU(0.15),
            nn.Conv1d(16, 32, kernel_size=2, padding=0),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(16, 8, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(8, 4, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(4, 2, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(0.15),
            nn.ConvTranspose1d(2, 1, kernel_size=2, stride=1, padding=0),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.squeeze(1)
        return x


class CNN_02(nn.Module):
    def __init__(self):
        super(CNN_02, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),  # input: 1 x 512, output: 16 x 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv1d(16, 32, 3, padding=1),  # input: 16 x 512, output: 32 x 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, 3, padding=1),  # input: 16 x 512, output: 32 x 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, 3, padding=1),  # input: 16 x 512, output: 32 x 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, 3, padding=1),  # input: 32 x 512, output: 16 x 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv1d(16, 1, 3, padding=1),  # input: 16 x 512, output: 1 x 512
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.squeeze(1)
        return x


class CNN_03(nn.Module):
    def __init__(self):
        super(CNN_03, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32768, 512),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.squeeze(1)
        return x
