import os
import shutil
from typing import Any, Iterable, Tuple, Union

import torch
import torch.nn as nn
from ignite.handlers.param_scheduler import LRScheduler
from thop import profile
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

from . import utils
from .optim.constraint import ChannelConstraint
from .optim.objective import (
    Accuracy,
    AccuracyFinetuned,
    Macs,
    MacsPenalty,
    Objective,
    ObjectiveContainer,
)
from .optim.optimizer import BinaryGAOptimizer, IntegerGAOptimizer, Optimizer
from .prune.pruner import ChannelPruner, Pruner, ResnetModulePruner
from .train.distillation import KDLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (1, 3, 32, 32)


def vgg_best(
    finetune: bool,
    mode: str,
    output_dir: str,
    dropout_decay: float = 0.0,
    wd_decay: float = 0.0,
    iterative: bool = False,
    distille: bool = False,
    reset_params: bool = False,
    **kwargs,
) -> nn.Module:
    """Performs pruning of the VGG16 to find best compromise between accuracy and model's 
    number of MACs. Optimization problem in this case is given by:

    max : accuracy_pruned / accuracy_orig + weight * (1 - MACs_pruned / MACs_orig)

    Args:
        finetune (bool): Determines whether finetune pruned model during one epoch before measuring 
            its accuracy.
        mode (str): Determines type of the optimization problem, one of ['int', 'binary'].
        output_dir (str): Path to the folder where pruned models will be stored.
        dropout_decay (float, optional): Reduction of dropout probability in 
            dropout layers during each step of the pruning. Defaults to 0.0.
        wd_decay (float, optional): Reduction of weight decay. Defaults to 0.0.
        iterative (bool, optional): Determines whether perform iterative pruning. Defaults to False.
        distille (bool, optional): Determines whether to use Knowledge Distillation for model 
            finetuning. Defaults to False.
        reset_params (bool, optional): Determines whether to reset weights' values after pruning, 
            i. e. train from scratch after pruning. Defaults to False.

    Returns:
        nn.Module: Pruned VGG16 model.
    """
    if mode not in ["int", "binary"]:
        raise ValueError(f"Invalid mode {mode}, currently supported modes are: ['int, 'binary']")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    model = utils.get_vgg16()
    names = [name for name, _ in utils.prunable_modules(model)]
    pruner = ChannelPruner(names, INPUT_SHAPE)
    i = 0

    while True:
        optim = _integer_GA(model, **kwargs) if mode == "int" else _binary_GA(model, **kwargs)
        objective = _objective_best(model, pruner, finetune, kwargs.get("weight", 1.0))
        constraint = ChannelConstraint(model, pruner)
        solution = optim.maximize(objective, constraint)

        kwargs["weight_decay"] = round(kwargs.get("weight_decay", 0.0001) - i * wd_decay, 8)

        model = pruner.prune(model, solution)
        model = _reduce_dropout(model, dropout_decay)
        model = utils.reset_params(model) if reset_params else model
        model = _train(model, utils.get_vgg16() if distille else None, 256, **kwargs)

        torch.save(model, os.path.join(output_dir, f"vgg_best_{i}.pth"))
        _save_solution(solution, os.path.join(output_dir, f"vgg_best_{i}.txt"))
        i += 1

        if not iterative or solution.fitness.values[0] <= kwargs.get("min_improve", 1.15):
            break

    return model


def vgg_constrained(
    finetune: bool,
    mode: str,
    bounds: Iterable,
    output_dir: str,
    dropout_decay: float = 0.0,
    wd_decay: float = 0.0,
    distille: bool = False,
    reset_params: bool = False,
    **kwargs,
) -> nn.Module:
    """Performs pruning of the VGG16 to iteratively prune model according to specified MACs 
    percentage upper bounds. MACs percentage upper bounds should be in range (0, 1) and define 
    maximum allowed percentage of MACs according to MACs of the original unpruned model in each
    step of the iterative pruining. Optimization problem in this case is given by:

    max : accuracy_pruned / accuracy_orig - weight * max(0, (MACs_pruned - B_t) / (MACs_orig - B_t))

    where B_t represents MACs percentage upper bound at t-th iteration of the pruning.

    Args:
        finetune (bool): Determines whether finetune pruned model during one epoch before measuring 
            its accuracy.
        mode (str): Determines type of the optimization problem, one of ['int', 'binary'].
        bounds (Iterable): MACs percentage upper bounds according to MACs of the original unpruned 
            model, each bound should be in range (0, 1).
        output_dir (str): Path to the folder where pruned models will be stored.
        dropout_decay (float, optional): Reduction of dropout probability in 
            dropout layers during each step of the pruning. Defaults to 0.0.
        wd_decay (float, optional): Reduction of weight decay. Defaults to 0.0.
        distille (bool, optional): Determines whether to use Knowledge Distillation for model 
            finetuning. Defaults to False.
        reset_params (bool, optional): Determines whether to reset weights' values after pruning, 
            i. e. train from scratch after pruning. Defaults to False.

    Returns:
        nn.Module: Pruned VGG16 model.
    """
    if mode not in ["int", "binary"]:
        raise ValueError(f"Invalid mode {mode}, currently supported modes are: ['int, 'binary']")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    model = utils.get_vgg16()
    names = [name for name, _ in utils.prunable_modules(model)]
    pruner = ChannelPruner(names, INPUT_SHAPE)
    orig_macs, _ = profile(model, inputs=(torch.randn(INPUT_SHAPE, device=DEVICE),), verbose=False)
    w = kwargs.get("weight", -1.0)

    # Iteratively prune model according to upper bounds
    for i, b in enumerate(bounds):
        optim = _integer_GA(model, **kwargs) if mode == "int" else _binary_GA(model, **kwargs)
        objective = _objective_constrained(model, pruner, finetune, orig_macs, b, w)
        constraint = ChannelConstraint(model, pruner)
        solution = optim.maximize(objective, constraint)

        kwargs["weight_decay"] = round(kwargs.get("weight_decay", 0.0001) - i * wd_decay, 8)

        model = pruner.prune(model, solution)
        model = _reduce_dropout(model, dropout_decay)
        model = utils.reset_params(model) if reset_params else model
        model = _train(model, utils.get_vgg16() if distille else None, 256, **kwargs)

        torch.save(model, os.path.join(output_dir, f"vgg_constrained_{b}.pth"))
        _save_solution(solution, os.path.join(output_dir, f"vgg_constrained_{b}.txt"))

    return model


def resnet_best(
    finetune: bool,
    mode: str,
    output_dir: str,
    iterative: bool = False,
    alternate: bool = True,
    wd_decay: float = 0.0,
    distille: bool = False,
    reset_params: bool = False,
    **kwargs,
) -> nn.Module:
    """Performs pruning of the ResNet56 to find best compromise between accuracy and model's 
    number of MACs. Optimization problem in this case is given by:

    max : accuracy_pruned / accuracy_orig + weight * (1 - MACs_pruned / MACs_orig)

    Args:
        finetune (bool): Determines whether finetune pruned model during one epoch before measuring 
            its accuracy.
        mode (str): Determines type of the optimization problem, one of ['int', 'binary'].
        output_dir (str): Path to the folder where pruned models will be stored.
        iterative (bool, optional): Determines whether perform iterative pruning. Defaults to False.
        alternate (bool, optional): Determines whether to alternatively perform channel and block 
            pruning. Defaults to True.
        wd_decay (float, optional): Reduction of weight decay. Defaults to 0.0.
        distille (bool, optional): Determines whether to use Knowledge Distillation for model 
            finetuning. Defaults to False.
        reset_params (bool, optional): Determines whether to reset weights' values after pruning, 
            i. e. train from scratch after pruning. Defaults to False.

    Returns:
        nn.Module: Pruned ResNet56 model.
    """
    if mode not in ["int", "binary"]:
        raise ValueError(f"Invalid mode {mode}, currently supported modes are: ['int, 'binary']")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    model = utils.get_resnet56()
    i = 0
    m_solution = None

    while True:
        # Channel pruning
        ch_names = [name for name, _ in utils.prunable_modules(model)]
        ch_pruner = ChannelPruner(ch_names, INPUT_SHAPE)

        optim = _integer_GA(model, **kwargs) if mode == "int" else _binary_GA(model, **kwargs)
        objective = _objective_best(model, ch_pruner, finetune, kwargs.get("weight", 1.0))
        constraint = ChannelConstraint(model, ch_pruner)
        ch_solution = optim.maximize(objective, constraint)

        # Block pruning
        if alternate:
            m_names = [n for n, m in model.named_modules() if type(m).__name__ == "BasicBlock"]
            m_pruner = ResnetModulePruner(m_names, "shortcut")

            optim = _module_GA(len(m_names), **kwargs)
            objective = _objective_best(model, m_pruner, finetune, kwargs.get("weight", 1.0))
            m_solution = optim.maximize(objective, None)

        kwargs["weight_decay"] = round(kwargs.get("weight_decay", 0.0001) - i * wd_decay, 8)

        model, solution = _choose_best(model, ch_solution, ch_pruner, m_solution, m_pruner)
        model = utils.reset_params(model) if reset_params else model
        model = _train(model, utils.get_resnet56() if distille else None, 128, **kwargs)

        torch.save(model, os.path.join(output_dir, f"resnet_best_{i}.pth"))
        _save_solution(solution, os.path.join(output_dir, f"resnet_best_{i}.txt"))
        i += 1

        if not iterative or solution.fitness.values[0] <= kwargs.get("min_improve", 1.15):
            break

    return model


def resnet_constrained(
    finetune: bool,
    mode: str,
    bounds: Iterable,
    output_dir: str,
    alternate: bool = True,
    wd_decay: float = 0.0,
    distille: bool = False,
    reset_params: bool = False,
    **kwargs,
) -> nn.Module:
    """Performs pruning of the ResNet56 to iteratively prune model according to specified MACs 
    percentage upper bounds. MACs percentage upper bounds should be in range (0, 1) and define 
    maximum allowed percentage of MACs according to MACs of the original unpruned model in each
    step of the iterative pruining. Optimization problem in this case is given by:

    max : accuracy_pruned / accuracy_orig - weight * max(0, (MACs_pruned - B_t) / (MACs_orig - B_t))

    where B_t represents MACs percentage upper bound at t-th iteration of the pruning.

    Args:
        finetune (bool): Determines whether finetune pruned model during one epoch before measuring 
            its accuracy.
        mode (str): Determines type of the optimization problem, one of ['int', 'binary'].
        bounds (Iterable): MACs percentage upper bounds according to MACs of the original unpruned 
            model, each bound should be in range (0, 1).
        output_dir (str): Path to the folder where pruned models will be stored.
        alternate (bool, optional): Determines whether to alternatively perform channel and block 
            pruning. Defaults to True.
        wd_decay (float, optional): Reduction of weight decay. Defaults to 0.0.
        distille (bool, optional): Determines whether to use Knowledge Distillation for model 
            finetuning. Defaults to False.
        reset_params (bool, optional): Determines whether to reset weights' values after pruning, 
            i. e. train from scratch after pruning. Defaults to False.

    Returns:
        nn.Module: Pruned ResNet56 model.
    """
    if mode not in ["int", "binary"]:
        raise ValueError(f"Invalid mode {mode}, currently supported modes are: ['int, 'binary']")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    model = utils.get_resnet56()
    orig_macs, _ = profile(model, inputs=(torch.randn(INPUT_SHAPE, device=DEVICE),), verbose=False)
    w = kwargs.get("weight", -1.0)
    m_solution = None

    # Iteratively prune model according to upper bounds
    for i, b in enumerate(bounds):
        # Channel pruning
        ch_names = [name for name, _ in utils.prunable_modules(model)]
        ch_pruner = ChannelPruner(ch_names, INPUT_SHAPE)

        optim = _integer_GA(model, **kwargs) if mode == "int" else _binary_GA(model, **kwargs)
        objective = _objective_constrained(model, ch_pruner, finetune, orig_macs, b, w)
        constraint = ChannelConstraint(model, ch_pruner)
        ch_solution = optim.maximize(objective, constraint)

        # Block pruning
        if alternate:
            m_names = [n for n, m in model.named_modules() if type(m).__name__ == "BasicBlock"]
            m_pruner = ResnetModulePruner(m_names, "shortcut")

            optim = _module_GA(len(m_names), **kwargs)
            objective = _objective_constrained(model, m_pruner, finetune, orig_macs, b, w)
            m_solution = optim.maximize(objective, None)

        kwargs["weight_decay"] = round(kwargs.get("weight_decay", 0.0001) - i * wd_decay, 8)

        model, solution = _choose_best(model, ch_solution, ch_pruner, m_solution, m_pruner)
        model = utils.reset_params(model) if reset_params else model
        model = _train(model, utils.get_resnet56() if distille else None, 128, **kwargs)

        torch.save(model, os.path.join(output_dir, f"resnet_constrained_{b}.pth"))
        _save_solution(solution, os.path.join(output_dir, f"resnet_constrained_{b}.txt"))

    return model


def _optimization_data() -> Tuple[Iterable, Iterable, Iterable]:
    train_loader, val_loader, test_loader = utils.cifar10_loaders(
        folder=os.path.join(os.getcwd(), "data", "cifar10"),
        batch_size=512,
        val_size=5000,
        train_transform=Compose([ToTensor()]),
        test_transform=Compose([ToTensor()]),
    )

    train_data = utils.loader_to_memory(train_loader, DEVICE)
    val_data = utils.loader_to_memory(val_loader, DEVICE)
    test_data = utils.loader_to_memory(test_loader, DEVICE)

    return train_data, val_data, test_data


def _train_data(batch_size) -> Tuple[Iterable, Iterable, Iterable]:
    train_loader, val_loader, test_loader = utils.cifar10_loaders(
        folder=os.path.join(os.getcwd(), "data", "cifar10"),
        batch_size=batch_size,
        val_size=5000,
        train_transform=Compose([RandomHorizontalFlip(p=0.5), RandomCrop(32, 4), ToTensor()]),
        test_transform=Compose([ToTensor()]),
    )

    return train_loader, val_loader, test_loader


def _objective_best(model: nn.Module, pruner: Pruner, finetune: bool, w: float) -> Objective:
    train_data, val_data, test_data = _optimization_data()
    orig_acc = utils.evaluate(model, test_data, DEVICE)
    orig_macs, _ = profile(model, inputs=(torch.randn(INPUT_SHAPE, device=DEVICE),), verbose=False)

    acc = (
        Accuracy(model, pruner, 1.0, val_data, orig_acc)
        if not finetune
        else AccuracyFinetuned(model, pruner, 1.0, train_data, val_data, len(train_data), orig_acc)
    )
    macs = Macs(model, pruner, orig_macs, w, in_shape=INPUT_SHAPE)

    return ObjectiveContainer(acc, macs)


def _objective_constrained(
    model: nn.Module, pruner: Pruner, finetune: bool, orig_macs: int, p: float, w: float
) -> Objective:
    train_data, val_data, test_data = _optimization_data()
    w = -1.0 * abs(w)
    orig_acc = utils.evaluate(model, test_data, DEVICE)

    acc = (
        Accuracy(model, pruner, 1.0, val_data, orig_acc)
        if not finetune
        else AccuracyFinetuned(model, pruner, 1.0, train_data, val_data, len(train_data), orig_acc)
    )
    macs = MacsPenalty(model, pruner, w, p, orig_macs, INPUT_SHAPE)

    return ObjectiveContainer(acc, macs)


def _integer_GA(model: nn.Module, **kwargs) -> Optimizer:
    bounds = [(0, len(module.weight) - 1) for _, module in utils.prunable_modules(model)]
    pop_size = kwargs.get("pop_size", 100)
    ind_size = len(bounds)

    return IntegerGAOptimizer(
        ind_size=ind_size,
        pop_size=pop_size,
        elite_num=kwargs.get("elite_num", int(0.1 * pop_size)),
        tourn_size=kwargs.get("tourn_size", int(0.1 * pop_size)),
        n_gen=kwargs.get("n_gen", 50),
        mutp=kwargs.get("mutp", 0.1),
        mut_indp=kwargs.get("mut_indp", 0.05),
        cx_indp=kwargs.get("cx_indp", 0.5),
        bounds=bounds,
    )


def _binary_GA(model: nn.Module, **kwargs) -> Optimizer:
    pop_size = kwargs.get("pop_size", 100)
    ind_size = sum([len(module.weight) for _, module in utils.prunable_modules(model)])

    return BinaryGAOptimizer(
        ind_size=ind_size,
        pop_size=pop_size,
        elite_num=kwargs.get("elite_num", int(0.1 * pop_size)),
        tourn_size=kwargs.get("tourn_size", int(0.1 * pop_size)),
        n_gen=kwargs.get("n_gen", 50),
        mutp=kwargs.get("mutp", 0.1),
        mut_indp=kwargs.get("mut_indp", 0.01),
        cx_indp=kwargs.get("cx_indp", 0.5),
    )


def _module_GA(ind_size: int, **kwargs) -> Optimizer:
    pop_size = kwargs.get("pop_size", 100)

    return BinaryGAOptimizer(
        ind_size=ind_size,
        pop_size=pop_size,
        elite_num=kwargs.get("elite_num", int(0.1 * pop_size)),
        tourn_size=kwargs.get("tourn_size", int(0.1 * pop_size)),
        n_gen=kwargs.get("n_gen", 50),
        mutp=kwargs.get("mutp", 0.1),
        mut_indp=kwargs.get("mut_indp", 0.01),
        cx_indp=kwargs.get("cx_indp", 0.5),
    )


def _train(model: nn.Module, teacher: nn.Module, batch_size: int, **kwargs) -> nn.Module:
    checkpoint = os.path.join(os.getcwd(), "tmp", "checkpoint")

    if os.path.exists(checkpoint):
        shutil.rmtree(checkpoint)

    lr = kwargs.get("lr", 0.01)
    momentum = kwargs.get("momentum", 0.9)
    weight_decay = kwargs.get("weight_decay", 0.0001)
    epochs = kwargs.get("epochs", 50)
    T = kwargs.get("T", 1.0)

    train, _, test = _train_data(batch_size)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss() if teacher is None else KDLoss(teacher, train, test, DEVICE, T)

    scheduler = (
        kwargs.get("lr_scheduler")(optimizer, **kwargs.get("lr_scheduler_params"))
        if "lr_scheduler" in kwargs
        else CosineAnnealingLR(optimizer, epochs)
    )
    lr_scheduler = LRScheduler(scheduler)

    _ = utils.train_ignite(
        model=model,
        train_set=train,
        test_set=test,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        checkpoint_path=checkpoint,
        lr_scheduler=lr_scheduler,
    )

    model_f = next(
        f
        for f in os.listdir(checkpoint)
        if os.path.isfile(os.path.join(checkpoint, f)) and os.path.splitext(f)[1] == ".pt"
    )
    model.load_state_dict(torch.load(os.path.join(checkpoint, model_f)))

    return model


def _reduce_dropout(model: nn.Module, do_decay: float) -> nn.Module:
    for dropout in [module for module in model.modules() if isinstance(module, nn.Dropout)]:
        dropout.p = max(0, round(dropout.p - do_decay, 4))

    return model


def _save_solution(solution: Any, out_f: str) -> None:
    with open(out_f, "a") as dest:
        dest.write(f"{','.join([str(x) for x in solution])}")


def _choose_best(
    model: nn.Module, ch_sol: Any, ch_pr: Pruner, m_sol: Any, m_pr: Pruner
) -> Tuple[nn.Module, Any]:
    pr = ch_pr if m_sol is None or ch_sol.fitness.values[0] > m_sol.fitness.values[0] else m_pr
    sol = ch_sol if m_sol is None or ch_sol.fitness.values[0] > m_sol.fitness.values[0] else m_sol

    return pr.prune(model, sol), sol
