import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple

import torch
import torch.nn as nn
from thop import profile
from torch.optim import SGD

from .. import utils
from ..prune.pruner import Pruner
from .cache import Cache

warnings.simplefilter("ignore", UserWarning)


class Objective(ABC):
    """Abstract class representing an objective function in optimization problems."""

    def __init__(self) -> None:
        """Ctor."""
        super().__init__()

    @abstractmethod
    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        """Computes value of the objective function for given solution.

        Given function allows users to model objective functions with arbitrary number of 
        dimensions, where number of elements in output tuple corresponds to the number of 
        dimensions of the function.

        Args:
            solution (Any): Solution produced by optimization algorithm to be evaluated.

        Returns:
            Tuple[float, ...]: Value of the objective function. Number of elements corresponds
                with number of dimensions of the objective function.
        """
        pass


class ObjectiveContainer(Objective):
    """Represents a container for modelling composite objective functions.
    
    Objective function represented by ObjectiveContainer is evaluated by summing individual
    objectives contained within it. All of the objectives must be of the same dimension.
    """

    def __init__(self, *objectives: Objective) -> None:
        """Ctor.

        Args:
            objectives (Objective): List of objective functions to be added to the container.
        """
        super().__init__()

        self._objs = objectives

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        obj_vals = [obj.evaluate(solution) for obj in self._objs]

        if not all(len(tup) == len(obj_vals[0]) for tup in obj_vals):
            raise ValueError("All objectives must be of same dimension")

        return tuple(map(sum, zip(*obj_vals)))


class ModelObjective(Objective):
    """Represents base class for objective functions related to a neural network pruning."""

    def __init__(self, model: nn.Module, pruner: Pruner) -> None:
        """Ctor.

        Args:
            model (nn.Module): Model to be pruned.
            pruner (Pruner): Pruner used for pruning the model.
        """
        super().__init__()

        self._model = model
        self._pruner = pruner

    def _model_device(self, model: nn.Module) -> str:
        return next(model.parameters()).device

    def _get_pruned_model(self, solution: Any) -> nn.Module:
        return Cache.get_pruned_model(self._model, self._pruner, solution)


class Accuracy(ModelObjective):
    """Represents objective function measuring neural network's accuracy after pruning.
    
    Value of the objective function is normalized according to accuracy of the original 
    unpruned neural network. Also weight can be specified when used in composite objective
    function i. e.: f = weight * (accuracy_pruned / accuracy_original).
    """

    def __init__(
        self, model: nn.Module, pruner: Pruner, weight: float, val_data: Iterable, orig_acc: float
    ) -> None:
        """Ctor.

        Args:
            model (nn.Module): Model to be pruned.
            pruner (Pruner): Pruner used for pruning the model.
            weight (float): Can be used to specify relative weight if used in composite function.
            val_data (Iterable): Validation set on which an accuracy will be measured.
            orig_acc (float): Accuracy of original unpruned model on the validation set.
        """
        super().__init__(model, pruner)

        self._weight = weight
        self._data = val_data
        self._orig_acc = orig_acc

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._get_pruned_model(solution)
        device = self._model_device(model)
        accuracy = utils.evaluate(model, self._data, device)

        return (self._weight * accuracy / self._orig_acc,)


class AccuracyFinetuned(ModelObjective):
    """Represents objective function measuring model's accuracy after pruning and finetuning.

    Before measuring pruned model's accuracy, model is finetuned on training set for 
    specified number of iterations. After that, model's accuracy is evaluated. Value 
    of the objective function is normalized according to accuracy of the original 
    unpruned neural network. Also weight can be specified when used in composite 
    objective function i. e.: f = weight * (accuracy_pruned / accuracy_original).
    """

    def __init__(
        self,
        model: nn.Module,
        pruner: Pruner,
        weight: float,
        train_data: Iterable,
        val_data: Iterable,
        iterations: int,
        orig_acc: float,
    ) -> None:
        """Ctor.

        Args:
            model (nn.Module): Model to be pruned.
            pruner (Pruner): Pruner used for pruning the model.
            weight (float): Can be used to specify relative weight if used in composite function.
            train_data (Iterable): Training set on which pruning model will be finetuned.
            val_data (Iterable): Validation set on which an accuracy will be measured.
            iterations (int): Number of finetuning iterations (total batches).
            orig_acc (float): Accuracy of original unpruned model on the validation set.
        """
        super().__init__(model, pruner)

        self._weight = weight
        self._train = train_data
        self._val = val_data
        self._iters = iterations
        self._orig_acc = orig_acc

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._get_pruned_model(solution)
        device = self._model_device(model)

        optim = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss()

        model = utils.train(model, self._train, device, optim, loss_fn, self._iters)
        accuracy = utils.evaluate(model, self._val, device)

        return (self._weight * accuracy / self._orig_acc,)


class Macs(ModelObjective):
    """Represents objective function measuring decrease in model's MACs after pruning.

    Value of the objective function is normalized according to original model's MACs. Also 
    weight can be specified when used in composite objective function. Final value of the 
    objective function is given by: f = weight * (1 - MACs_pruned / MACs_original).
    """

    def __init__(
        self,
        model: nn.Module,
        pruner: Pruner,
        orig_macs: int,
        weight: float,
        in_shape: Tuple[int, ...],
    ) -> None:
        """Ctor.

        Args:
            model (nn.Module): Model to be pruned.
            pruner (Pruner): Pruner used for pruning the model.
            orig_macs (int): Number of MACs of original unpruned model.
            weight (float): Can be used to specify relative weight if used in composite function.
            in_shape (Tuple[int, ...]): Model's input shape.
        """
        super().__init__(model, pruner)

        self._orig_macs = orig_macs
        self._weight = weight
        self._in_shape = in_shape

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._get_pruned_model(solution)
        device = self._model_device(model)
        in_tensor = torch.randn(self._in_shape, device=device)
        macs, _ = profile(model, inputs=(in_tensor,), verbose=False)

        return (self._weight * (1.0 - macs / self._orig_macs),)


class MacsPenalty(ModelObjective):
    """Represents penalty function for exceeding maximum allowed number of MACs.
    
    Number of MACs of pruned model is normalized according to number of MACs of the original 
    unpruned model. Also weight can be specified when used in composite objective function. 
    Final value of the objective function is given by:
    f = weight * max(0, (MACs_pruned - B) / (MACs_original - B)) 
    where B is maximum allowed number of MACs for the penalty function.
    """

    def __init__(
        self,
        model: nn.Module,
        pruner: Pruner,
        weight: float,
        p: float,
        orig_macs: int,
        in_shape: Tuple[int, ...],
    ) -> None:
        """Ctor.

        Args:
            model (nn.Module): Model to be pruned.
            pruner (Pruner): Pruner used for pruning the model.
            weight (float): Can be used to specify relative weight if used in composite function.
            p (float): Value between (0, 1) specifying maximum allowed portion of MACs according 
                to the original MACs.
            orig_macs (int): Number of MACs of original unpruned model.
            in_shape (Tuple[int, ...]): Model's input shape.
        """
        super().__init__(model, pruner)

        self._weigh = weight
        self._p = p
        self._orig_macs = orig_macs
        self._input_shape = in_shape

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._get_pruned_model(solution)
        device = self._model_device(model)
        in_tensor = torch.randn(self._input_shape, device=device)
        macs, _ = profile(model, inputs=(in_tensor,), verbose=False)

        # To scale the penalty to [0, 1], we need to divide current penalty by maximum possible
        # penalty, i. e.: max(0, macs - orig_macs * p) / (orig_macs - orig_macs * p).
        penalty = max(0.0, macs - self._orig_macs * self._p)
        penalty_scaled = penalty / (self._orig_macs - self._orig_macs * self._p)
        penalty_weighted = self._weigh * penalty_scaled

        return (penalty_weighted,)


class SizePenalty(ModelObjective):
    """Represents penalty function for exceeding maximum or minimum of allowed model size.
    
    Size is computed by dividing total number of pruned model's weights by total number of 
    original model's weights. i. e.: f = weight * (num_weights_pruned / num_weights_orig).
    """

    def __init__(
        self,
        model: nn.Module,
        pruner: Pruner,
        weight: float,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        """Ctor.

        Args:
            model (nn.Module): Model to be pruned.
            pruner (Pruner): Pruner used for pruning the model.
            weight (float): Can be used to specify relative weight if used in composite function.
            lower_bound (float): Value between (0, upper_bound) specifying minimum allowed size.
            upper_bound (float): Value between (lower_bound, 1) specifying maximum allowed size.
        """
        super().__init__(model, pruner)

        self._weight = weight
        self._lbound = lower_bound
        self._ubound = upper_bound
        self._orig_size = self._count_params(self._model)

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._get_pruned_model(solution)
        size = self._count_params(model) / self._orig_size

        if size < self._lbound:
            return (self._weight * (self._lbound - size),)
        elif size > self._ubound:
            return (self._weight * (size - self._ubound),)
        else:
            return (0.0,)

    def _count_params(self, model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
