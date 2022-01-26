from typing import Iterable, Tuple
import pyacc.pruning.prune as prune
from pyacc.pruning.model.abstract import ModelPruner, Strategy
from pyacc.pruning.scoring.abstract import Scoring
from torch import nn


class GlobalPruner(ModelPruner):
    def __init__(
        self, prune_types: Tuple[type], exclude_modules: Iterable[str], factor: float
    ) -> None:
        super().__init__(prune_types, exclude_modules)

        self.__factor = factor

    def prune(
        self, model: nn.Module, scoring: Scoring, strategy: Strategy, shrink_model: bool
    ) -> None:
        layers = [module for name, module in model.named_modules() if self._prunable(name, module)]

        if strategy == Strategy.Structured:
            params = [layer + ("weight", 0) for layer in layers]
            prune.global_structured(params, self.__factor, scoring)
        elif strategy == Strategy.Unstructured:
            params = [layer + ("weight",) for layer in layers]
            prune.global_unstructured(params, self.__factor, scoring)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        
        if shrink_model:
            [self._shrink_model(model, name for name, _ in model.named_modules())]
