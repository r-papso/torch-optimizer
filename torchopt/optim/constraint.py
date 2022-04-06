from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from ..prune.pruner import Pruner
from .cache import Cache


class Constraint(ABC):
    """Abstract class representing a constraint in optimization problems."""

    def __init__(self) -> None:
        """Ctor."""
        super().__init__()

    @abstractmethod
    def feasible(self, solution: Any) -> bool:
        """Determines if solution produced by optimization algorithm satisfies given constraint.

        Args:
            solution (Any): Solution produced by optimization algorithm to be evaluated.

        Returns:
            bool: True if solution satisfies given constraint, False otherwise.
        """
        pass


class ConstraintContainer(Constraint):
    """Represents a container for modelling optimization problems with multiple conditions.

    Constraint represented by ConstraintContainer is considered feasible (i. e. function feasible
    returns True) if and only if all of the constraints within the container are satisfied.
    """

    def __init__(self, *constraints: Constraint) -> None:
        """Ctor.

        Args:
            constraints (Constraint): List of constraints to be added to the container.
        """
        super().__init__()

        self._constrs = constraints

    def feasible(self, solution: Any) -> bool:
        return all(constr.feasible(solution) for constr in self._constrs)


class ChannelConstraint(Constraint):
    """Constrain for checking a validity of pruning mask.
    
    Constraint checks if solution, which represents pruning mask, will produce valid pruned 
    model (neural network). Pruned model is invalid if any of it's weight tensors are fully 
    pruned, i. e. weight tensor's first dimension is less than 1.
    """

    def __init__(self, model: nn.Module, pruner: Pruner) -> None:
        """Ctor.

        Args:
            model (nn.Module): Model to be pruned.
            pruner (Pruner): Pruner used for pruning the model.
        """
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
