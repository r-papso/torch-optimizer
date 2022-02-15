import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader

warnings.simplefilter("ignore", UserWarning)


class Objective(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        pass


class Penalty(Objective):
    def __init__(self, previous: Objective) -> None:
        super().__init__()

        self._prev = previous

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        return self._prev.evaluate(model)


class Accuracy(Objective):
    def __init__(self, val_loader: DataLoader, device: str) -> None:
        super().__init__()

        self._loader = val_loader
        self._device = device

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in self._loader:
                inputs = inputs.to(self._device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        return (correct / total,)


class MacsPenalty(Penalty):
    def __init__(
        self,
        previous: Objective,
        weight: float,
        p: float,
        orig_macs: int,
        in_shape: Tuple[int, ...],
    ) -> None:
        super().__init__(previous)

        self._weigh = weight
        self._p = p
        self._orig_macs = orig_macs
        self._input_shape = in_shape

    def evaluate(self, model: nn.Module) -> Tuple[float, ...]:
        obj_vals = super().evaluate(model)
        macs, _ = profile(model, inputs=(torch.randn(self._input_shape),), verbose=False)
        penalty = self._weigh * max(0.0, macs - self._orig_macs * self._p)
        return tuple([obj_val + penalty for obj_val in obj_vals])
