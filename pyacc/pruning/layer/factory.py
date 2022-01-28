from pyacc.pruning.layer.abstract import (
    AdaptivePool2dReducer,
    BatchNorm2dReducer,
    Conv2dReducer,
    FlattenReducer,
    IdentityReducer,
    LayerPruner,
    LinearReducer,
    Pool2dReducer,
    Reducer,
)
from pyacc.pruning.layer.batchnorm2d import BatchNorm2dPruner
from pyacc.pruning.layer.conv import ConvPruner
from pyacc.pruning.layer.linear import LinearPruner
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
        return pruner() if pruner is not None else None

    @classmethod
    def register_pruner(cls, module_type: type, pruner_type: type) -> None:
        cls._pruners[module_type] = pruner_type


class ReducerFactory:
    _reducers = {
        nn.Conv2d: Conv2dReducer,
        nn.Linear: LinearReducer,
        nn.BatchNorm2d: BatchNorm2dReducer,
        nn.Flatten: FlattenReducer,
        nn.MaxPool2d: Pool2dReducer,
        nn.AvgPool2d: Pool2dReducer,
        nn.AdaptiveMaxPool2d: AdaptivePool2dReducer,
        nn.AdaptiveAvgPool2d: AdaptivePool2dReducer,
    }

    @classmethod
    def get(cls, module_type: type) -> Reducer:
        reducer = cls._reducers.get(module_type, None)
        return reducer() if reducer is not None else IdentityReducer()

    @classmethod
    def register(cls, module_type: type, reducer_type: type) -> None:
        cls._reducers[module_type] = reducer_type
