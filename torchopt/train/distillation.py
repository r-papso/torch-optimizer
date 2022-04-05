from copy import deepcopy
from torch import nn, Tensor
from torch.nn import functional as F

from .loader import DataLoaderWrapper


class KDLoss(nn.Module):
    def __init__(self, teacher: nn.Module, loader: DataLoaderWrapper, device: str) -> None:
        super().__init__()

        self._loader = loader
        self._teacher = self._init_teacher(teacher)
        self._device = device
        self._kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        data, labels = self._loader.last_batch()
        data, labels = data.to(self._device), labels.to(self._device)

        assert all(labels == targets)

        preds = self._teacher(data)
        preds_log = F.log_softmax(preds, dim=1)
        inputs_log = F.log_softmax(inputs, dim=1)

        return self._kl_loss(inputs_log, preds_log)

    def _init_teacher(self, teacher: nn.Module) -> nn.Module:
        teacher_cpy = deepcopy(teacher)

        for param in teacher_cpy.parameters():
            param.requires_grad = False

        return teacher_cpy
