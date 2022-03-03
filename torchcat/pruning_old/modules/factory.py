from .reducer import *
from torch import nn


class ReducerFactory:
    _reducers = {
        nn.Conv1d: ConvReducer,
        nn.Conv2d: ConvReducer,
        nn.Conv3d: ConvReducer,
        nn.BatchNorm1d: BatchNormReducer,
        nn.BatchNorm2d: BatchNormReducer,
        nn.BatchNorm3d: BatchNormReducer,
        nn.MaxPool1d: PoolReducer,
        nn.AvgPool1d: PoolReducer,
        nn.MaxPool2d: PoolReducer,
        nn.AvgPool2d: PoolReducer,
        nn.MaxPool3d: PoolReducer,
        nn.AvgPool3d: PoolReducer,
        nn.AdaptiveMaxPool1d: AdaptivePoolReducer,
        nn.AdaptiveAvgPool1d: AdaptivePoolReducer,
        nn.AdaptiveMaxPool2d: AdaptivePoolReducer,
        nn.AdaptiveAvgPool2d: AdaptivePoolReducer,
        nn.AdaptiveMaxPool3d: AdaptivePoolReducer,
        nn.AdaptiveAvgPool3d: AdaptivePoolReducer,
        nn.Linear: LinearReducer,
        nn.Flatten: FlattenReducer,
    }

    @classmethod
    def get(cls, module_type: type) -> Reducer:
        reducer = cls._reducers.get(module_type, None)
        return reducer() if reducer is not None else FixedOpReducer()

    @classmethod
    def register(cls, module_type: type, reducer_type: type) -> None:
        cls._reducers[module_type] = reducer_type
