import torch
from torch import nn as nn
from torch.nn import init as init


class LSTM_01(nn.Module):
    def __init__(self):
        super(LSTM_01, self).__init__()

        in_dim = 512
        hid_dim = 256
        out_dim = 512
        num_layers = 1

        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(0.1),
        )
        self.lstm_layer = nn.LSTM(hid_dim, hid_dim, num_layers, batch_first=True)
        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor):
        y = self.in_layer(x)
        y, _ = self.lstm_layer(y)
        y = self.out_layer(y)
        return y
