from torch import nn
from pruning.layer.abstract import LayerPruner
from pruning.layer.batchnorm2d import BatchNorm2dPruner
from pruning.layer.conv2d import Conv2dPruner
from pruning.layer.linear import LinearPruner


class LayerPrunerFactory:
    _pruners = {nn.Conv2d: Conv2dPruner, nn.Linear: LinearPruner, nn.BatchNorm2d: BatchNorm2dPruner}

    @classmethod
    def get_pruner(cls, module_type: type) -> LayerPruner:
        pruner = cls._pruners.get(module_type, None)

        if pruner is None:
            raise ValueError(f"{LayerPrunerFactory} -> Unsupported module type: {module_type}.")

        return pruner()

    @classmethod
    def register_pruner(cls, module_type: type, pruner_type: type) -> None:
        cls._pruners[module_type] = pruner_type
