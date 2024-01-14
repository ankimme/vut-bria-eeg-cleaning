from torch import nn as nn


from models.fcnn import FCNN_01


class ModelFactory:
    @staticmethod
    def FCNN_01() -> nn.Module:
        return FCNN_01()
