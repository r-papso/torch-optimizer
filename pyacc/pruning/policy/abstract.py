from abc import ABC, abstractmethod

from torch import nn


class Policy(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_fraction(self, layer: nn.Module) -> float:
        pass
