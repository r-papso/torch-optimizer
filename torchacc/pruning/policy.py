from abc import ABC, abstractmethod
from typing import Dict

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


class DictPolicy(Policy):
    def __init__(self, module_dict: Dict[str, float], model: nn.Module) -> None:
        super().__init__()

        self.__dict = module_dict
        self.__model = model

    def get_fraction(self, module: nn.Module) -> float:
        name = next(n for n, m in self.__model.named_modules() if m is module)
        return self.__dict[name]
