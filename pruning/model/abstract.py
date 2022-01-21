from typing import List
from abc import ABC, abstractmethod
from torch import nn

from strategy.abstract import Strategy

prunable_types = (nn.Conv2d, nn.Linear)


class Pruner(ABC):
    def __init__(self, strategy: Strategy, remove_channels: bool) -> None:
        super().__init__()

        self.__strategy = strategy
        self.__remove_channels = remove_channels

    def prune(self, model: nn.Module, exclude_layers: List[str] = None) -> None:
        # TODO reimplement
        new_model = nn.Sequential()

        for layer in model.modules():
            if self._prunable(layer):
                mask = self.get_mask(layer)
            else:
                new_model.add_module()

    def _prunable(self, module: nn.Module) -> bool:
        return isinstance(module, prunable_types)

