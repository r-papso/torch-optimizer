from abc import ABC, abstractmethod

import torch.nn as nn


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
