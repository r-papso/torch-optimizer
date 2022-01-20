from typing import List
import numpy as np

from abc import ABC, abstractmethod
from torch import nn


class Scoring(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_score(self, layer: nn.Module) -> np.ndarray:
        pass

    @abstractmethod
    def get_scores(self, layers: List[nn.Module]) -> List[np.ndarray]:
        pass
