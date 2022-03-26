from datetime import datetime
from typing import Callable, Iterable, Tuple

import ignite.metrics as metrics
import torch
import torch.nn as nn
from torch import Tensor
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.handlers.param_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10


def cifar10_loaders(
    folder: str,
    batch_size: int,
    val_size: int,
    train_transform: Callable,
    test_transform: Callable,
) -> Tuple[DataLoader, ...]:
    train_set = CIFAR10(download=True, root=folder, transform=train_transform, train=True)
    test_set = CIFAR10(download=False, root=folder, transform=test_transform, train=False)

    idxs = list(range(len(train_set)))
    split = len(train_set) - val_size
    train_idxs, val_idxs = idxs[:split], idxs[split:]

    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def loader_to_memory(data_loader: DataLoader, device: str) -> Iterable[Tuple[Tensor, Tensor]]:
    data = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        data.append((inputs, labels))

    return data


def train_ignite(
    model: nn.Module,
    train_set: Iterable[Tuple[Tensor, Tensor]],
    test_set: Iterable[Tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    loss_fn: Callable,
    epochs: int,
    checkpoint_path: str = None,
    lr_scheduler: LRScheduler = None,
) -> dict:
    metric_dict = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = model.to(device)

    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=device, output_transform=_custom_output_transform
    )

    val_metrics = {"accuracy": metrics.Accuracy(), "loss": metrics.Loss(loss_fn)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    for name, metric in val_metrics.items():
        metric.attach(trainer, name)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_train, metric_dict, optimizer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_test, evaluator, test_set, metric_dict)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_time)

    if lr_scheduler is not None:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)

    if checkpoint_path is not None:
        checkpoint = _create_checkpoint(model, trainer, checkpoint_path)
        evaluator.add_event_handler(Events.COMPLETED, checkpoint)

    trainer.run(train_set, max_epochs=epochs)
    return metric_dict


def train(
    model: nn.Module,
    data: Iterable[Tuple[Tensor, Tensor]],
    device: str,
    optimizer: Optimizer,
    loss_fn: Callable,
    iterations: int,
) -> nn.Module:
    model = model.train().to(device)
    iters = 0

    while iters < iterations:
        for inputs, labels in data:
            if inputs.device != device:
                inputs = inputs.to(device)

            if labels.device != device:
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            iters += 1
            if iters == iterations:
                break

    return model


def evaluate(model: nn.Module, data: Iterable[Tuple[Tensor, Tensor]], device: str) -> float:
    model = model.eval().to(device)
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in data:
            if inputs.device != device:
                inputs = inputs.to(device)

            if labels.device != device:
                labels = labels.to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return correct / total


def _custom_output_transform(x, y, y_pred, loss):
    return {"y": y, "y_pred": y_pred, "loss": loss.item(), "criterion_kwargs": {}}


def _log_train(engine: Engine, metric_dict: dict, optimizer: Optimizer) -> None:
    epoch = engine.state.epoch
    metrics = engine.state.metrics
    acc = metrics["accuracy"]
    loss = metrics["loss"]
    time = datetime.now().strftime("%H:%M:%S")
    lr = optimizer.param_groups[0]["lr"]

    metric_dict[epoch] = {"train_acc": acc, "train_loss": loss}
    print(
        f"{time} - Epoch: {epoch:04d} Train accuracy: {acc:.4f} Train loss: {loss:.4f} LR: {lr:.8f}"
    )


def _log_test(engine: Engine, evaluator: Engine, test_set: Iterable, metric_dict: dict) -> None:
    evaluator.run(test_set)

    epoch = engine.state.epoch
    metrics = evaluator.state.metrics
    acc = metrics["accuracy"]
    loss = metrics["loss"]
    time = datetime.now().strftime("%H:%M:%S")

    metric_dict[epoch].update({"test_acc": acc, "test_loss": loss})
    print(f"{time} - Epoch: {epoch:04d} Test accuracy:  {acc:.4f} Test loss:  {loss:.4f}")


def _log_time(engine: Engine) -> None:
    name = engine.last_event_name.name
    time = engine.state.times[name]
    print(f"{name} took {time:.4f} seconds")


def _create_checkpoint(model: nn.Module, trainer: Engine, path: str) -> Checkpoint:
    to_save = {"model": model}
    return Checkpoint(
        to_save,
        path,
        n_saved=1,
        filename_prefix="best",
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),
        greater_or_equal=True,
    )
