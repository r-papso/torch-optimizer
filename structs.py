from dataclasses import dataclass

import torch


@dataclass
class ParamMask:
    name: str
    mask: torch.Tensor
