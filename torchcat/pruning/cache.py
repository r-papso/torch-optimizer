from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import nn

from .pruner import Pruner


class Cache:
    _instance = None

    def __init__(self, model: nn.Module, pruner: Pruner) -> None:
        self._model = model
        self._pruner = pruner
        self._cached_model = None

    @classmethod
    def get_cache(cls, model: nn.Module, pruner: Pruner) -> Cache:
        if cls._instance is None:
            cls._instance = cls(model, pruner)

        if cls._instance._model is not model or cls._instance._pruner is not pruner:
            cls._instance = cls(model, pruner)

        return cls._instance

    def get_cached_model(self, solution: Any) -> nn.Module:
        if self._cached_model is not None and self._cached_model[0] is solution:
            return self._cached_model[1]

        if self._cached_model is not None:
            del self._cached_model[1]

        model_cpy = deepcopy(self._model)
        model_cpy = self._pruner.prune(model_cpy, solution)
        self._cached_model = (solution, model_cpy)

        return self._cached_model[1]
