from __future__ import annotations

from typing import Any

from torch import nn

from ..prune.pruner import Pruner
from . import utils


class Cache:
    __model = None
    __solution = None

    @classmethod
    def get_pruned_model(cls, model: nn.Module, pruner: Pruner, solution: Any) -> nn.Module:
        if solution is not None and solution is cls.__solution:
            return cls.__model

        if solution is not None and solution is not cls.__solution:
            cached_model = cls.__model
            cls.__model = None
            del cached_model

        cls.__solution = solution
        cls.__model = utils.prune_model(model, pruner, solution)

        return cls.__model
