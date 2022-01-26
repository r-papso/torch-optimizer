from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Tuple
from pyacc.pruning.layer.factory import LayerPrunerFactory

from pyacc.pruning.scoring.abstract import Scoring
from torch import nn


class Strategy(Enum):
    Structured = 0
    Unstructured = 1


class ModelPruner(ABC):
    def __init__(self, prune_modules: Iterable[str]) -> None:
        super().__init__()

        self._modules = prune_modules

    @abstractmethod
    def prune(
        self, model: nn.Module, scoring: Scoring, strategy: Strategy, shrink_model: bool
    ) -> None:
        pass

    def _shrink_model(self, model: nn.Module, module_name: str) -> None:
        layer = model.get_submodule(module_name)
        pruner = LayerPrunerFactory.get_pruner(type(layer))

        if pruner is None or not pruner.prunable_by_mask(layer):
            return

        channels = pruner.prune_by_mask(layer)
        names = [name for name, _ in model.named_modules()]
        idx = names.index(module_name)
        names = names[idx + 1 :]

        for name in names:
            layer = model.get_submodule(name)
            pruner = LayerPrunerFactory.get_pruner(type(layer))

            if pruner is not None:
                if not pruner.prune_by_channels(layer, channels):
                    break
