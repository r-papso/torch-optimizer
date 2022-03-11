from copy import deepcopy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from thop import profile

from .pruner import Pruner

from .. import utils

warnings.simplefilter("ignore", UserWarning)


class Objective(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        pass


class ObjectiveContainer(Objective):
    def __init__(self, *objectives: Objective) -> None:
        super().__init__()

        self._objs = objectives

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        obj_vals = [obj.evaluate(solution) for obj in self._objs]

        if not all(len(tup) == len(obj_vals[0]) for tup in obj_vals):
            raise ValueError("All objectives must be of same dimension")

        return tuple(map(sum, zip(*obj_vals)))


class ModelObjective(Objective):
    def __init__(self, model: nn.Module, pruner: Pruner) -> None:
        super().__init__()

        self._model = model
        self._pruner = pruner

    def _prune_model(self, solution: Any) -> nn.Module:
        model_cpy = deepcopy(self._model)
        model_cpy = self._pruner.prune(model_cpy, solution)
        return model_cpy


class Accuracy(ModelObjective):
    def __init__(self, model: nn.Module, pruner: Pruner, val_data: Iterable) -> None:
        super().__init__(model, pruner)

        self._data = val_data

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._prune_model(solution)
        accuracy = utils.evaluate(model, self._data)
        del model
        return (accuracy,)


class MacsPenalty(ModelObjective):
    def __init__(
        self,
        model: nn.Module,
        pruner: Pruner,
        weight: float,
        p: float,
        orig_macs: int,
        in_shape: Tuple[int, ...],
    ) -> None:
        super().__init__(model, pruner)

        self._weigh = weight
        self._p = p
        self._orig_macs = orig_macs
        self._input_shape = in_shape

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._prune_model(solution)
        device = next(model.parameters()).device
        in_tensor = torch.randn(self._input_shape, device=device)
        macs, _ = profile(model, inputs=(in_tensor,), verbose=False)
        del model

        # To scale the penalty to [0, 1], we need to divide current penalty by maximum possible
        # penalty, i. e.: max(0, macs - orig_macs * p) / (orig_macs - orig_macs * p).
        penalty = max(0.0, macs - self._orig_macs * self._p)
        penalty_scaled = penalty / (self._orig_macs - self._orig_macs * self._p)
        penalty_weighted = self._weigh * penalty_scaled

        return (penalty_weighted,)


class LatencyPenalty(ModelObjective):
    def __init__(
        self,
        model: nn.Module,
        pruner: Pruner,
        weight: float,
        p: float,
        orig_time: float,
        in_shape: Tuple[int, ...],
        n_iters: int,
    ) -> None:
        super().__init__(model, pruner)

        self._weigh = weight
        self._p = p
        self._orig_time = orig_time
        self._in_shape = in_shape
        self._n_iters = n_iters

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._prune_model(solution)
        times = self.profile(model)
        avg_time = np.average(times)
        del model

        # To scale the penalty to [0, 1], we need to divide current penalty by maximum possible
        # penalty, i. e.: max(0, time - orig_time * p) / (orig_time - orig_time * p).
        penalty = max(0.0, avg_time - self._orig_time * self._p)
        penalty_scaled = penalty / (self._orig_time - self._orig_time * self._p)
        penalty_weighted = self._weigh * penalty_scaled

        return (penalty_weighted,)

    def profile(self, model: nn.Module) -> List[float]:
        device = next(model.parameters()).device
        times = []
        model.eval()

        for _ in range(self._n_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            in_tensor = torch.randn(self._in_shape, device=device)

            start.record()
            _ = model(in_tensor)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        return times


class Macs(ModelObjective):
    def __init__(
        self,
        model: nn.Module,
        pruner: Pruner,
        orig_macs: int,
        weight: float,
        in_shape: Tuple[int, ...],
    ) -> None:
        super().__init__(model, pruner)

        self._orig_macs = orig_macs
        self._weight = weight
        self._in_shape = in_shape

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._prune_model(solution)
        device = next(model.parameters()).device
        in_tensor = torch.randn(self._in_shape, device=device)
        macs, _ = profile(model, inputs=(in_tensor,), verbose=False)
        del model

        return (self._weight * (1.0 - macs / self._orig_macs),)


class LeakyAccuracy(ModelObjective):
    def __init__(
        self,
        model: nn.Module,
        pruner: Pruner,
        a: float,
        b: float,
        t: float,
        val_data: Iterable[Tuple[Tensor, Tensor]],
    ) -> None:
        super().__init__(model, pruner)

        self._a = a
        self._b = b
        self._t = t
        self._data = val_data

    def evaluate(self, solution: Any) -> Tuple[float, ...]:
        model = self._prune_model(solution)
        accuracy = utils.evaluate(model, self._data)
        del model
        return (min(self._a * (accuracy - self._t), self._b * (accuracy - self._t)),)
