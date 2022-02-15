import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader

warnings.simplefilter("ignore", UserWarning)


class Evaluator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self, model: nn.Module) -> float:
        pass


class EvaluatorDecorator(Evaluator):
    def __init__(self, wrappee: Evaluator) -> None:
        super().__init__()

        self._wrappee = wrappee

    def evaluate(self, model: nn.Module) -> float:
        return self._wrappee.evaluate(model) if self._wrappee is not None else 0.0


class AccuracyEvaluator(EvaluatorDecorator):
    def __init__(self, wrappee: Evaluator, val_loader: DataLoader, device: str) -> None:
        super().__init__(wrappee)

        self._loader = val_loader
        self._device = device

    def evaluate(self, model: nn.Module) -> float:
        prev_evals = super().evaluate(model)

        if prev_evals < 0.0:
            return prev_evals

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in self._loader:
                inputs = inputs.to(self._device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        return correct / total


class ChannelEvaluator(EvaluatorDecorator):
    def __init__(self, wrappee: Evaluator) -> None:
        super().__init__(wrappee)

    def evaluate(self, model: nn.Module) -> float:
        prev_evals = super().evaluate(model)

        if prev_evals < 0.0:
            return prev_evals

        for module in model.modules():
            weight = getattr(module, "weight", None)
            if weight is not None and any(dim <= 0 for dim in weight.shape):
                return -1.0

        return 0.0


class MacsEvaluator(EvaluatorDecorator):
    def __init__(
        self, wrappee: Evaluator, p: float, orig_macs: int, input_shape: Tuple[int, ...],
    ) -> None:
        super().__init__(wrappee)

        self._p = p
        self._orig_macs = orig_macs
        self._input_shape = input_shape

    def evaluate(self, model: nn.Module) -> float:
        prev_evals = super().evaluate(model)

        if prev_evals < 0.0:
            return prev_evals

        macs, _ = profile(model, inputs=(torch.randn(self._input_shape),), verbose=False)
        return 0.0 if macs <= self._orig_macs * self._p else -1.0
