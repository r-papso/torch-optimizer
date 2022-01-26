from typing import Iterable, Tuple

import pyacc.pruning.prune as prune
from pyacc.pruning.model.abstract import ModelPruner, Strategy
from pyacc.pruning.policy.abstract import Policy
from pyacc.pruning.scoring.abstract import Scoring
from torch import nn


class LocalPruner(ModelPruner):
    def __init__(self, prune_modules: Iterable[str], policy: Policy) -> None:
        super().__init__(prune_modules)

        self.__policy = policy

    def prune(
        self, model: nn.Module, scoring: Scoring, strategy: Strategy, shrink_model: bool
    ) -> None:
        for name, module in model.named_modules():
            if name in self._modules:
                factor = self.__policy.get_fraction(module)

                if strategy == Strategy.Structured:
                    prune.local_structured(module, "weight", factor, scoring, 0)
                elif strategy == Strategy.Unstructured:
                    prune.local_unstructured(module, "weight", factor, scoring)
                else:
                    raise ValueError(f"Invalid strategy: {strategy}")

                if shrink_model:
                    self._shrink_model(model, name)
