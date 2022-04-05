from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .loader import DataLoaderWrapper


class KDLoss(nn.Module):
    def __init__(
        self,
        teacher: nn.Module,
        train_loader: DataLoaderWrapper,
        test_loader: DataLoaderWrapper,
        device: str,
    ) -> None:
        super().__init__()

        self._train = train_loader
        self._test = test_loader
        self._teacher = self._init_teacher(teacher)
        self._device = device
        self._kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self._last_test, self._last_train = None, None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        batch = None

        if self._last_train is not self._train.cahced_batch():
            batch = self._train.cahced_batch()
            self._last_train = batch
        elif self._last_test is not self._test.cahced_batch():
            batch = self._test.cahced_batch()
            self._last_test = batch
        else:
            raise ValueError("Both train and test batches were already used")

        data, labels = batch[0].to(self._device), batch[1].to(self._device)

        assert torch.all(labels == targets)

        preds = self._teacher(data)
        preds_log = F.log_softmax(preds, dim=1)
        inputs_log = F.log_softmax(inputs, dim=1)

        return self._kl_loss(inputs_log, preds_log)

    def _init_teacher(self, teacher: nn.Module) -> nn.Module:
        teacher_cpy = deepcopy(teacher)

        for param in teacher_cpy.parameters():
            param.requires_grad = False

        return teacher_cpy
