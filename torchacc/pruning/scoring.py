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
        mask = getattr(layer, f"{name}_mask", None)
        score = torch.abs(torch.float_power(param, self.__n))

        if mask is not None:
            score = torch.where(mask == False, torch.Tensor([float("inf")]), score)

        return score
