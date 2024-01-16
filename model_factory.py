from torch import nn as nn


from models.fcnn import FCNN_01, FCNN_02
from models.lstm import LSTM_01
from models.cnn import CNN_01, CNN_02, CNN_03


class ModelFactory:
    @staticmethod
    def FCNN_01() -> nn.Module:
        return FCNN_01()

    @staticmethod
    def FCNN_02() -> nn.Module:
        return FCNN_02()

    @staticmethod
    def LSTM_01() -> nn.Module:
        return LSTM_01()

    @staticmethod
    def CNN_01() -> nn.Module:
        return CNN_01()

    @staticmethod
    def CNN_02() -> nn.Module:
        return CNN_02()

    @staticmethod
    def CNN_03() -> nn.Module:
        return CNN_03()
