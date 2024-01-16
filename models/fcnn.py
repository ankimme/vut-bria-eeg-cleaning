import torch
from torch import nn as nn
from torch.nn import init as init


class FCNN_01(torch.nn.Module):
    def __init__(self):
        super(FCNN_01, self).__init__()

        in_dim = 512
        hid_dim = 1024
        out_dim = 512

        # define network
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim, hid_dim),
        )
        self.out_head = nn.Linear(hid_dim, out_dim)

        # initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode="fan_out")
                init.constant_(layer.bias, 0)

        init.kaiming_normal_(self.out_head.weight, mode="fan_out")
        init.constant_(self.out_head.bias, 0)

    def forward(self, x: torch.Tensor):
        y = self.net(x)
        y = self.out_head(y)
        return y


class FCNN_02(torch.nn.Module):
    def __init__(self):
        super(FCNN_02, self).__init__()

        in_dim = 512
        hid_dim = 1024
        out_dim = 512

        # define network
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.1),
        )
        self.out_head = nn.Linear(hid_dim, out_dim)

        # initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode="fan_out")
                init.constant_(layer.bias, 0)

        init.kaiming_normal_(self.out_head.weight, mode="fan_out")
        init.constant_(self.out_head.bias, 0)

    def forward(self, x: torch.Tensor):
        y = self.net(x)
        y = self.out_head(y)
        return y
