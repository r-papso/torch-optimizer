import numpy as np

from abc import ABC, abstractmethod
from torch import nn


class Scoring(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_score(self, layer: nn.Module) -> np.ndarray:
        pass
