from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Iterable, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning.dependency import _get_module_type


class Pruner(ABC):
    """Base class for neural network pruning implementations."""

    def __init__(self) -> None:
        """Ctor."""
        super().__init__()

    @abstractmethod
    def prune(self, model: nn.Module, mask: Any) -> nn.Module:
        """Performs structured pruning on a model according to specified mask.

        After pruning all unnecessary parameters are physically removed from the model, 
        thus reducing the total number of parameters contained in the model.

        Args:
            model (nn.Module): Model to be pruned.
            mask (Any): Mask defining pruning structure.

        Returns:
            nn.Module: Pruned model.
        """
        pass


class ModulePruner(Pruner):
    """Pruner implementation performing pruning in module-wise fashion.
    
    ModulePruner prunes whole neural network's modules. A module can be single layer
    of the network or group of layers, also called blocks. Only modules contained in 
    nn.Sequential parent module can be pruned.
    """

    def __init__(self, names: Iterable[str]) -> None:
        """Ctor.

        Args:
            names (Iterable[str]): Names of modules to be pruned.
        """
        super().__init__()

        self._names = names

    def prune(self, model: nn.Module, mask: Any) -> nn.Module:
        assert len(mask) == len(self._names), "Mask's length must be equal to names length"
        assert all(x in [0, 1] for x in mask), "Module pruner supports only pruning by binary masks"

        for name in [name for sign, name in zip(mask, self._names) if not sign]:
            model = self._prune_module(model, name)

        return model

    def _prune_module(self, model: nn.Module, name: str) -> nn.Module:
        names = name.split(".")
        sequential = model.get_submodule(".".join(names[:-1]))

        if not isinstance(sequential, nn.Sequential):
            raise ValueError("Pruning modules outside of sequential containers is not supported")

        parent = model.get_submodule(".".join(names[:-2])) if len(names) > 2 else model
        filtered = OrderedDict(
            [(ch_name, ch) for ch_name, ch in sequential.named_children() if ch_name != names[-1]]
        )
        setattr(parent, names[-2], nn.Sequential(filtered))

        return model


class ResnetModulePruner(ModulePruner):
    """Implementation of ModulePruner specific to Resnet architectures.

    This implementation takes care of the shortcut connections contained in Resnet 
    architectures. If a pruned block contains the shortcut connection, this connection 
    is preserved in the model to preserve its functional properties.
    """

    def __init__(self, names: Iterable[str], shortcut_name: str) -> None:
        """Ctor.

        Args:
            names (Iterable[str]): Names of modules to be pruned.
            shortcut_name (str): Name of the module representing shortcut connection.
        """
        super().__init__(names)

        self._shortcut_name = shortcut_name

    def _prune_module(self, model: nn.Module, name: str) -> nn.Module:
        sc_name = self._shortcut_name

        # Shortcut module is not present in the sequential module
        if not any(sc_name in ch_name for ch_name, _ in model.get_submodule(name).named_children()):
            return super()._prune_module(model, name)

        shortcut = model.get_submodule(f"{name}.{sc_name}")

        # Shortcut connection is empty
        if isinstance(shortcut, nn.Sequential) and len(list(shortcut.children())) == 0:
            return super()._prune_module(model, name)

        # Shortcut connection contains only identity module
        if isinstance(shortcut, nn.Identity):
            return super()._prune_module(model, name)

        p_name, ch_name = name.rsplit(".", 1)
        setattr(model.get_submodule(p_name), ch_name, shortcut)
        return model


class ChannelPruner(Pruner):
    """Pruner implementation performing pruning in channel-wise fashion.

    ChannelPruner prunes individual filters/neurons from the neural network's layers. It accepts 
    two types of masks:

    1 - binary which can contain {0, 1} only. Length of binary mask must be the same as 
    the total number of filters/neurons in pruned layers. 0 at the i-th position within 
    the mask means that i-th filter/neuron will be pruned from the model.

    2 - integer which can contain integers only. Length of integer mask must be the same
    as the total number of pruned layers. I-th element at mask defines number of fiters/
    neurons to be pruned at the i-th pruned layer. Specific filters/neurons are determined
    according to the sum of their elements' absolute values (L1 strategy).
    """

    def __init__(self, module_names: Iterable[str], input_shape: Tuple[int, ...],) -> None:
        """Ctor.

        Args:
            module_names (Iterable[str]): Names of modules to be pruned.
            input_shape (Tuple[int, ...]): Model's input shape.
        """
        super().__init__()

        self._input_shape = input_shape
        self._names = module_names

    def prune(self, model: nn.Module, mask: Any) -> nn.Module:
        device = next(model.parameters()).device
        example_input = torch.randn(self._input_shape, device=device)
        DG = tp.DependencyGraph()
        DG = DG.build_dependency(model, example_inputs=example_input)

        for name, idxs in zip(self._names, self._get_indexes(model, mask)):
            if len(idxs) > 0:
                module = model.get_submodule(name)
                optype = _get_module_type(module)

                pruning_func = tp.DependencyGraph.HANDLER[optype][1]
                pruning_plan = DG.get_pruning_plan(module, pruning_func, idxs)

                _ = pruning_plan.exec()

        return model

    def _get_indexes(self, model: nn.Module, mask: Any) -> Iterable[Iterable[int]]:
        result = []

        # Binary mask
        if sum(len(model.get_submodule(name).weight) for name in self._names) == len(mask):
            assert all(x in [0, 1] for x in mask), "Binary mask must contain only {0, 1}"
            length = 0

            for name in self._names:
                w_length = len(model.get_submodule(name).weight)
                module_mask = mask[length : length + w_length]
                idxs = [i for i in range(len(module_mask)) if not module_mask[i]]
                result.append(idxs)
                length += w_length

        # Integer mask
        elif len(self._names) == len(mask):
            assert all(isinstance(x, int) for x in mask), "Integer mask must contain integers only"

            for name, amount in zip(self._names, mask):
                strategy = tp.strategy.L1Strategy()
                idxs = strategy(model.get_submodule(name).weight, amount=amount)
                result.append(idxs)

        # Unknown mask type
        else:
            raise ValueError("Invalid mask type")

        return result
