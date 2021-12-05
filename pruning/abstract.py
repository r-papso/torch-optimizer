import numpy as np

from typing import List
from abc import ABC, abstractmethod
from torch import nn

from pipeline.pipeline_context import PipelineContext
from pipeline.abstract import PipelineEntity
from strategy.abstract import Strategy
from dataobject.layer_mask import LayerMask


class Pruner(PipelineEntity):
    def __init__(
        self, context: PipelineContext, strategy: Strategy, n_steps: int, remove_channels: bool,
    ) -> None:
        super().__init__(context)

        self.__strategy = strategy
        self.__n_steps = n_steps
        self.__remove_channels = remove_channels

    @abstractmethod
    def get_fraction(self, layer: nn.Module = None) -> float:
        pass

    @abstractmethod
    def get_masks(self, layers: List[nn.Module]) -> List[LayerMask]:
        pass

    def prune(self) -> None:
        model = self._context.model()
        new_model = nn.Sequential()

        for layer in model.modules():
            if self._prunable(layer):
                mask = self.get_mask(layer)
            else:
                new_model.add_module()

    def n_steps(self) -> int:
        return self.__n_steps

    def _prunable(self, module: nn.Module) -> bool:
        return isinstance(module, (nn.Conv2d, nn.Linear))

