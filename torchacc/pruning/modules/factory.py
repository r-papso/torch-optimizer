from .reducer import *
from torch import nn


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
