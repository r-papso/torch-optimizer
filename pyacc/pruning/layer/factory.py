from pruning.layer.abstract import LayerPruner
from pruning.layer.batchnorm2d import BatchNorm2dPruner
from pruning.layer.conv import ConvPruner
from pruning.layer.linear import LinearPruner
from torch import nn


class LayerPrunerFactory:
    _pruners = {
        nn.Conv1d: ConvPruner,
        nn.Conv2d: ConvPruner,
        nn.Conv3d: ConvPruner,
        nn.Linear: LinearPruner,
        nn.BatchNorm2d: BatchNorm2dPruner,
    }

    @classmethod
    def get_pruner(cls, module_type: type) -> LayerPruner:
        pruner = cls._pruners.get(module_type, None)

        if pruner is None:
            raise ValueError(f"{LayerPrunerFactory} -> Unsupported module type: {module_type}.")

        return pruner()

    @classmethod
    def register_pruner(cls, module_type: type, pruner_type: type) -> None:
        cls._pruners[module_type] = pruner_type
