from abc import ABC, abstractmethod

from torch import nn


class Policy(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_fraction(self, module: nn.Module) -> float:
        pass


class ConstantPolicy(Policy):
    def __init__(self, fraction: float) -> None:
        super().__init__()

        self.__fraction = fraction

    def get_fraction(self, module: nn.Module) -> float:
        return self.__fraction
