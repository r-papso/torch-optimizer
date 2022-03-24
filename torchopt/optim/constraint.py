from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from ..prune.pruner import Pruner
from .cache import Cache


class Constraint(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def feasible(self, solution: Any) -> bool:
        pass


class ConstraintContainer(Constraint):
    def __init__(self, *constraints: Constraint) -> None:
        super().__init__()

        self._constrs = constraints

    def feasible(self, solution: Any) -> bool:
        return all(constr.feasible(solution) for constr in self._constrs)


class ChannelConstraint(Constraint):
    def __init__(self, model: nn.Module, pruner: Pruner) -> None:
        super().__init__()

        self._pruner = pruner
        self._model = model

    def feasible(self, solution: Any) -> bool:
        model = Cache.get_pruned_model(self._model, self._pruner, solution)

        for module in model.modules():
            weight = getattr(module, "weight", None)

            if weight is not None and any(dim <= 0 for dim in weight.shape):
                return False

        return True
