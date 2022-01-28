from abc import ABC, abstractmethod

import torch
from torch import nn


class Scoring(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_score(self, layer: nn.Module, name: str) -> torch.Tensor:
        pass


class LnScoring(Scoring):
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

    def get_score(self, layer: nn.Module, name: str) -> torch.Tensor:
        param = getattr(layer, name)
        return torch.abs(torch.float_power(param, self.__n))
