from abc import ABC, abstractmethod
from typing import Iterable, Tuple
from torch import Tensor

import torch.nn as nn

from .. import utils


class Constraint(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def feasible(self, model: nn.Module) -> bool:
        pass


class ConstraintContainer(Constraint):
    def __init__(self, *constraints: Constraint) -> None:
        super().__init__()

        self._constrs = constraints

    def feasible(self, model: nn.Module) -> bool:
        return all(constr.feasible(model) for constr in self._constrs)


class ChannelConstraint(Constraint):
    def __init__(self) -> None:
        super().__init__()

    def feasible(self, model: nn.Module) -> bool:
        for module in model.modules():
            weight = getattr(module, "weight", None)
            if weight is not None and any(dim <= 0 for dim in weight.shape):
                return False

        return True


class AccuracyConstraint(Constraint):
    def __init__(self, t: float, val_data: Iterable[Tuple[Tensor, Tensor]]) -> None:
        super().__init__()

        self._t = t
        self._data = val_data

    def feasible(self, model: nn.Module) -> bool:
        accuracy = utils.evaluate(model, self._data)
        return accuracy > self._t
