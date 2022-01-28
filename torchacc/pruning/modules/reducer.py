import itertools
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import torch
from torch import nn

from . import utils


class Reducer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        pass

    @abstractmethod
    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        pass


class ReducerBase(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        self._before_mask_reduce(module)
        module.zero_grad()

        with torch.no_grad():
            channel_mask = utils.channel_mask(module, "weight")

            w_slices = utils.create_slices(module, "weight", channel_mask, 0)
            b_slices = utils.create_slices(module, "bias", channel_mask, 0)

            utils.reduce_parameter(module, "weight", w_slices)
            utils.reduce_parameter(module, "bias", b_slices)

        self._after_mask_reduce(module, channel_mask)
        return self._mask_reduce_result(module, channel_mask)

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        self._befor_input_reduce(module, input)
        module.zero_grad()

        with torch.no_grad():
            dim = self._input_dependent_dim()
            slices = utils.create_slices(module, "weight", input[dim], dim)
            utils.reduce_parameter(module, "weight", slices)

        self._after_input_reduce(module, input)
        return self._input_reduce_result(module, input)

    @abstractmethod
    def _mask_reduce_result(
        self, module: nn.Module, channel_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        pass

    @abstractmethod
    def _input_reduce_result(
        self, module: nn.Module, input: Tuple[Iterable[int], ...]
    ) -> Tuple[Iterable[int], ...]:
        pass

    @abstractmethod
    def _input_dependent_dim(self) -> int:
        pass

    def _before_mask_reduce(self, module: nn.Module) -> None:
        pass

    def _befor_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        pass

    def _after_mask_reduce(self, module: nn.Module, channel_mask: Iterable[bool]) -> None:
        pass

    def _after_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        pass


class Conv2dReducer(ReducerBase):
    def __init__(self) -> None:
        super().__init__()

    def _mask_reduce_result(
        self, module: nn.Module, channel_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        if (out_shape := getattr(module, "out_shape", None)) is not None:
            h_out, w_out = [True] * out_shape[-2], [True] * out_shape[-1]
        else:
            h_out, w_out = None, None

        return (None, channel_mask, h_out, w_out)

    def _input_reduce_result(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        return None

    def _input_dependent_dim(self) -> int:
        return 1

    def _before_mask_reduce(self, module: nn.Module) -> None:
        assert isinstance(module, nn.Conv2d), f"Invalid layer type: {type(module)}."

    def _befor_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        assert isinstance(module, nn.Conv2d), f"Invalid layer type: {type(module)}."

    def _after_mask_reduce(self, module: nn.Module, channel_mask: Iterable[bool]) -> None:
        setattr(module, "out_channels", sum(channel_mask))
        utils.set_out_shape(module, 1, sum(channel_mask))

    def _after_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        setattr(module, "in_channels", sum(input[1]))


class LinearReducer(ReducerBase):
    def __init__(self) -> None:
        super().__init__()

    def _mask_reduce_result(
        self, module: nn.Module, channel_mask: Iterable[bool]
    ) -> Tuple[Iterable[bool], ...]:
        if (out_shape := getattr(module, "out_shape", None)) is not None:
            out = [[True] * out_dim for out_dim in out_shape.tolist()]
            out[-1] = channel_mask
        else:
            out = [None, channel_mask]

        return tuple(out)

    def _input_reduce_result(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        return None

    def _input_dependent_dim(self) -> int:
        return -1

    def _before_mask_reduce(self, module: nn.Module) -> None:
        assert isinstance(module, nn.Linear), f"Invalid layer type: {type(module)}."

    def _befor_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        assert isinstance(module, nn.Linear), f"Invalid layer type: {type(module)}."

    def _after_mask_reduce(self, module: nn.Module, channel_mask: Iterable[bool]) -> None:
        setattr(module, "out_features", sum(channel_mask))
        utils.set_out_shape(module, -1, sum(channel_mask))

    def _after_input_reduce(self, module: nn.Module, input: Tuple[Iterable[bool], ...]) -> None:
        setattr(module, "in_features", sum(input[-1]))


class BatchNorm2dReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, nn.BatchNorm2d), f"Invalid layer type: {type(module)}."

        module.zero_grad()
        with torch.no_grad():
            slices = (input[1],)
            param_names = ["running_mean", "running_var", "weight", "bias"]

            for name in param_names:
                utils.reduce_parameter(module, name, slices)

        module.num_features = sum(input[1])
        utils.set_out_shape(module, 1, sum(input[1]))

        return input


class FlattenReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(module, nn.Flatten), f"Invalid layer type: {type(module)}."

        start = module.start_dim
        end = module.end_dim if module.end_dim != -1 else len(input)
        dim_masks = input[start:end]

        assert all(mask is not None for mask in dim_masks)

        flattened = [all(vals) for vals in itertools.product(*dim_masks)]
        utils.set_out_shape(module, start, sum(flattened))

        return input[:start] + (flattened,) + input[end:]


class Pool2dReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(
            module, (nn.MaxPool2d, nn.AvgPool2d)
        ), f"Invalid layer type: {type(module)}."

        if (out_shape := getattr(module, "out_shape", None)) is not None:
            h_out, w_out = [True] * out_shape[-2], [True] * out_shape[-1]
        else:
            h_out, w_out = None, None

        utils.set_out_shape(module, 1, sum(input[1]))
        return (input[0], input[1], h_out, w_out)


class AdaptivePool2dReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        assert isinstance(
            module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)
        ), f"Invalid layer type: {type(module)}."

        out_size = module.output_size
        h_out_size = out_size[0] if isinstance(out_size, tuple) else out_size
        w_out_size = out_size[1] if isinstance(out_size, tuple) else out_size

        if (out_shape := getattr(module, "out_shape", None)) is not None:
            h_out_buff, w_out_buff = [True] * out_shape[-2], [True] * out_shape[-1]
        else:
            h_out_buff, w_out_buff = None, None

        h_out = [True] * h_out_size if h_out_size is not None else h_out_buff
        w_out = [True] * w_out_size if w_out_size is not None else w_out_buff

        utils.set_out_shape(module, 1, sum(input[1]))
        return (input[0], input[1], h_out, w_out)


class IdentityReducer(Reducer):
    def __init__(self) -> None:
        super().__init__()

    def reduce_by_mask(self, module: nn.Module) -> Tuple[Iterable[bool], ...]:
        raise ValueError(f"Reduction by mask is not supported.")

    def reduce_by_input(
        self, module: nn.Module, input: Tuple[Iterable[bool], ...]
    ) -> Tuple[Iterable[bool], ...]:
        for i, dim_mask in enumerate([d for d in input if d is not None]):
            utils.set_out_shape(module, i, sum(dim_mask))

        return input
