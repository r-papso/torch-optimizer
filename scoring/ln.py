import numpy as np

from torch import nn

from scoring.abstract import Scoring


class LnScoring(Scoring):
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

    def get_score(self, layer: nn.Module) -> np.ndarray:
        return np.abs(np.power(layer.weight.detach().numpy(), self.__n))
