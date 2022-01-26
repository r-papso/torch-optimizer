from abc import ABC, abstractmethod

from pyacc.pruning.scoring.abstract import Scoring
from torch import nn


class ModelPruner(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prune(self, model: nn.Module, scoring: Scoring, strategy: str, shrink_model: bool) -> None:
        pass
