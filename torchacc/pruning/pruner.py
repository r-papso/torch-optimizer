from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning.dependency import _get_module_type


class Pruner(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prune(self, model: nn.Module, individual: Any) -> nn.Module:
        pass


class ModulePruner(Pruner):
    def __init__(self, names: Iterable[str]) -> None:
        super().__init__()

        self._names = names

    def prune(self, model: nn.Module, individual: Any) -> nn.Module:
        assert len(individual) == len(
            self._names
        ), "Individual's length must be equal to names length"

        for name in [name for sign, name in zip(individual, self._names) if not sign]:
            model = self._prune_module(model, name)

        return model

    def _prune_module(self, model: nn.Module, name: str) -> nn.Module:
        names = name.split(".")
        sequential = model.get_submodule(".".join(names[:-1]))

        if not isinstance(sequential, nn.Sequential):
            raise ValueError("Pruning modules outside of sequential containers is not supported")

        parent = model.get_submodule(".".join(names[:-2])) if len(names) > 2 else model
        filtered = [ch for ch_name, ch in sequential.named_children() if ch_name != names[-1]]
        setattr(parent, names[-2], nn.Sequential(*filtered))

        return model


class ResnetModulePruner(ModulePruner):
    def __init__(self, names: Iterable[str]) -> None:
        super().__init__(names)

    def _prune_module(self, model: nn.Module, name: str) -> nn.Module:
        if "downsample" in [ch_name for ch_name, _ in model.get_submodule(name).named_children()]:
            downsample = model.get_submodule(f"{name}.downsample")
            p_name, ch_name = name.rsplit(".", 1)
            setattr(model.get_submodule(p_name), ch_name, downsample)
            return model
        else:
            return super()._prune_module(model, name)


class ChannelPruner(Pruner):
    def __init__(
        self, channel_map: Dict[str, Tuple[int, int]], input_shape: Tuple[int, ...],
    ) -> None:
        super().__init__()

        self._channel_map = channel_map
        self._input_shape = input_shape

    def prune(self, model: nn.Module, individual: Any) -> nn.Module:
        assert len(individual) == sum(
            v[1] for v in self._channel_map.values()
        ), "Individual's length must be equal to number of channels in channel_map"

        device = next(model.parameters()).device
        example_input = torch.randn(self._input_shape, device=device)
        DG = tp.DependencyGraph()
        DG = DG.build_dependency(model, example_inputs=example_input)

        for module_name, (start, lenght) in self._channel_map.items():
            module_mask = individual[start : start + lenght]

            if not all(module_mask):
                module = model.get_submodule(module_name)
                optype = _get_module_type(module)

                pruning_func = tp.DependencyGraph.HANDLER[optype][1]
                pruning_idxs = [i for i in range(len(module_mask)) if not module_mask[i]]
                pruning_plan = DG.get_pruning_plan(module, pruning_func, pruning_idxs)

                _ = pruning_plan.exec()

        return model
