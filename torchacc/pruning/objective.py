import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from thop import profile

warnings.simplefilter("ignore", UserWarning)


class Objective(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        pass


class ObjectiveContainer(Objective):
    def __init__(self, *objectives: Objective) -> None:
        super().__init__()

        self._objs = objectives

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        obj_vals = [obj.evaluate(model) for obj in self._objs]

        if not all(len(tup) == len(obj_vals[0]) for tup in obj_vals):
            raise ValueError("All objectives must be of same dimension")

        return tuple(map(sum, zip(*obj_vals)))


class Accuracy(Objective):
    def __init__(self, val_data: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__()

        self._data = val_data

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        model.eval()
        correct, total = 0, 0
        device = next(model.parameters()).device

        with torch.no_grad():
            for inputs, labels in self._data:
                if inputs.device != device:
                    inputs = inputs.to(device)

                if labels.device != device:
                    labels = labels.to(device)

                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        return (correct / total,)


class MacsPenalty(Objective):
    def __init__(self, weight: float, p: float, orig_macs: int, in_shape: Tuple[int, ...],) -> None:
        super().__init__()

        self._weigh = weight
        self._p = p
        self._orig_macs = orig_macs
        self._input_shape = in_shape

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        device = next(model.parameters()).device
        in_tensor = torch.randn(self._input_shape, device=device)
        macs, _ = profile(model, inputs=(in_tensor,), verbose=False)

        # To scale the penalty to [0, 1], we need to divide current penalty by maximum possible
        # penalty, i. e.: max(0, macs - orig_macs * p) / (orig_macs - orig_macs * p).
        penalty = max(0.0, macs - self._orig_macs * self._p)
        penalty_scaled = penalty / (self._orig_macs - self._orig_macs * self._p)
        penalty_weighted = self._weigh * penalty_scaled

        return (penalty_weighted,)
