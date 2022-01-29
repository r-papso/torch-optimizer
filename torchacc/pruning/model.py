from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable

from torch import nn

from . import prune
from .modules.factory import ReducerFactory
from .policy import Policy
from .scoring import Scoring


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
        reducer = ReducerFactory.get(type(layer))
        out = reducer.reduce_by_mask(layer)

        if out is None or all([mask is None or all(mask) for mask in out]):
            return

        names = [name for name, _ in model.named_modules()]
        idx = names.index(module_name)
        names = names[idx + 1 :]

        for name in names:
            layer = model.get_submodule(name)
            reducer = ReducerFactory.get(type(layer))
            out = reducer.reduce_by_input(layer, out)

            if out is None or all([mask is None or all(mask) for mask in out]):
                break


class GlobalPruner(ModelPruner):
    def __init__(self, prune_modules: Iterable[str], factor: float) -> None:
        super().__init__(prune_modules)

        self.__factor = factor

    def prune(
        self, model: nn.Module, scoring: Scoring, strategy: Strategy, shrink_model: bool
    ) -> None:
        layers = [module for name, module in model.named_modules() if name in self._modules]

        if strategy == Strategy.Structured:
            params = [layer + ("weight", 0) for layer in layers]
            prune.global_structured(params, self.__factor, scoring)
        elif strategy == Strategy.Unstructured:
            params = [layer + ("weight",) for layer in layers]
            prune.global_unstructured(params, self.__factor, scoring)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        if shrink_model:
            [self._shrink_model(model, name) for name, _ in model.named_modules()]


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
