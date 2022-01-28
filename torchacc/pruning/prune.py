import numpy as np
import torch

from typing import Iterable, Tuple
from torch import nn

from .scoring import Scoring


class _PruningHook:
    def __init__(self, name: str) -> None:
        self.__name = name

    def __call__(self, module: nn.Module, inputs: Tuple[torch.Tensor]) -> None:
        setattr(module, self.__name, apply_mask(module, self.__name))


def local_unstructured(layer: nn.Module, name: str, factor: float, scoring: Scoring) -> None:
    dim = getattr(layer, name).dim() - 1
    _pruning([(layer, name, dim)], factor, scoring)


def local_structured(
    layer: nn.Module, name: str, factor: float, scoring: Scoring, dim: int
) -> None:
    _pruning([(layer, name, dim)], factor, scoring)


def global_unstructured(
    params: Iterable[Tuple[nn.Module, str]], factor: float, scoring: Scoring
) -> None:
    dims = [getattr(layer, name).dim() - 1 for layer, name in params]
    params = [param + (dim) for param, dim in zip(params, dims)]
    _pruning(params, factor, scoring)


def global_structured(
    params: Iterable[Tuple[nn.Module, str, int]], factor: float, scoring: Scoring
) -> None:
    _pruning(params, factor, scoring)


def apply_mask(layer: nn.Module, name: str) -> torch.Tensor:
    mask = getattr(layer, f"{name}_mask")
    orig = getattr(layer, f"{name}_orig")
    pruned = mask * orig
    return pruned


def remove(layer: nn.Module, name: str) -> None:
    weight = apply_mask(layer, name)

    # Remove mask and orig from layer
    del layer._buffers[f"{name}_mask"]
    del layer._parameters[f"{name}_orig"]

    # Remove forward hook
    del_key = next(
        k for k, hook in layer._forward_pre_hooks.items() if isinstance(hook, _PruningHook)
    )
    del layer._forward_pre_hooks[del_key]

    # Set layer's weight
    layer.register_parameter(name, weight)


def _pruning(params: Iterable[Tuple[nn.Module, str, int]], factor: float, scoring: Scoring) -> None:
    masks = _get_masks(params, factor, scoring)
    masks = [_combine_masks(layer, name, mask) for (layer, name, _), mask in zip(params, masks)]
    [_set_mask(layer, name, mask) for (layer, name, _), mask in zip(params, masks)]


def _get_masks(
    params: Iterable[Tuple[nn.Module, str, int]], factor: float, scoring: Scoring
) -> Iterable[torch.Tensor]:
    scores = _get_flattened_scores(params, scoring)
    sorted_scores = sorted(scores, key=lambda tup: tup[0])

    pruned_fractions = [_get_pruned_fraction(layer, name) for layer, name, _ in params]
    p_actual = sum(pruned_fractions) / len(pruned_fractions)
    p = int((len(sorted_scores) - len(sorted_scores) * p_actual) * factor)

    masks = [torch.ones_like(getattr(layer, name), dtype=torch.bool) for layer, name, _ in params]

    for _, list_idx, slices in sorted_scores[0:p]:
        masks[list_idx][slices] = False

    return masks


def _get_flattened_scores(
    params: Iterable[Tuple[nn.Module, str, int]], scoring: Scoring
) -> Iterable[Tuple[float, int, slice]]:
    scores = [scoring.get_score(layer, name) for layer, name, _ in params]
    result = []

    for i, ((layer, name, dim), score) in enumerate(zip(params, scores)):
        layer_param = getattr(layer, name)

        if dim + 1 == len(layer_param.shape):
            # Unstructured
            score_agg = score
        else:
            # Structured
            dims_to_sum = tuple(range(dim + 1, len(score.shape)))
            # Divide score sum by its size to make it comparable across
            # whole model in case of global pruning
            score_agg = score.sum(dim=dims_to_sum) / score.shape[dim + 1 :].numel()

        # Iterate over all elements in score_agg
        for j in range(score_agg.size().numel()):
            # Get index to score_agg tensor from flattened index i
            idx_tuple = np.unravel_index(j, score_agg.shape)
            slices = [slice(None)] * score.ndim

            for k, idx in enumerate(idx_tuple):
                slices[k] = idx

            score_val = score_agg[idx_tuple].item()
            result.append((score_val, i, tuple(slices)))

    return result


def _get_pruned_fraction(layer: nn.Module, name: str) -> float:
    mask = getattr(layer, name, None)
    fraction = mask[mask == 0].size().numel() / mask.size().numel() if mask is not None else 0.0
    return fraction


def _combine_masks(layer: nn.Module, name: str, mask: torch.Tensor) -> torch.Tensor:
    actual = getattr(layer, f"{name}_mask", None)
    new_mask = actual * mask if actual is not None else mask
    return new_mask


def _set_mask(layer: nn.Module, name: str, mask: torch.Tensor) -> None:
    # Remove old and add new mask into layer
    layer._buffers.pop(f"{name}_mask", None)
    layer.register_buffer(f"{name}_mask", mask)

    # Register weight_orig into layer's parameters if not registered, yet
    if f"{name}_orig" not in layer._parameters:
        orig = layer.get_parameter(name)
        layer.register_parameter(f"{name}_orig", orig)
        del layer._parameters[name]

    # As we removed weight from layer's parameters, we need to set it
    # as attribute manually
    setattr(layer, name, apply_mask(layer, name))

    # Register forward hook if not registered, yet
    if not any([isinstance(hook, _PruningHook) for hook in layer._forward_pre_hooks.values()]):
        layer.register_forward_pre_hook(_PruningHook(name))
