from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.module import Module
from torch.utils.data.dataset import Dataset


class PipelineContext:
    def __init__(self, model: nn.Module, data: Dataset) -> None:
        self.__model = model
        self.__data = data

    def model(self) -> nn.Module:
        return self.__model

    def data(self) -> Dataset:
        return self.__data
