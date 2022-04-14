# Torch optimizer
Torch optimizer is a Python library for optimizing PyTorch models using techniques of neural network pruning. Neural network pruning can be formulated as an optimization problem to determine best subset of set of network's weights, i. e.:

**Maximize:** Accuracy(model(W • M)) <br>
**Subject to:** Resource<sub>j</sub>(model(W • M)) <= Budget<sub>j</sub>

where W are model's weights, M is binary mask with size |M| = |W|, resource can be any resource we want to reduce (e. g. FLOPs, MACs, latency, model size, ...) and budget is our desired upper bound of the resource we want to reduce. 

Library provides several functionalities to facilitate solving given optimization problem.

## Features

### Objective functions

[Objective](./torchopt/optim/objective.py) module provides common interface for modelling optimization objective functions. Objective function of arbitrary optimization problem can be created by implementing the interface. Module also provides several implementations of objective function in context of neural network pruning to evaluate pruned neural net's performance and efficiency, such as accuracy or relative decrease of MACs (Multiply–accumulate operations) according to number of MACs of original unpruned net.

### Constraints

[Constraint](./torchopt/optim/constraint.py) module provides common interface for modelling optimization constraints. Constraint of an arbitrary optimization problem can be created by the interface implementation. For neural network pruning purposes, a constraint that checks validity of the pruning (i. e. no layer can contain empty weight tensor after pruning) is provided.

### Optimization algorithms

[Optimizer](./torchopt/optim/optimizer.py) module contains common interface for an optimization algortihm implementations. Module also contains implementation of Genetic algorithm (GA) meta-heuristic. Two implementations of GA are provided: 1. to solve integer optimization problems and 2. to solve binary optimization problems. Detailed description of GA implementations can be found in the module.

### Pruning

[Pruner](./torchopt/prune/pruner.py) module provides basic functionality for structured neural network pruning. Structured pruning can be performed in different levels of granularity. For channel pruning, where individual filters / neurons are pruned, a channel pruner is provided. For module level pruning, where individual layers or blocks of the network can be pruned, library provides module pruner implementation.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Torch optimizer.

```bash
pip install torch-optim
```

## Usage

One can train their own PyTorch model on arbitrary dataset and then use the library functionalities to perform structured pruning. Here is a simple example:

```python
import torch

from torchopt import utils
from torchopt.prune.pruner import ChannelPruner
from torchopt.optim.optimizer import IntegerGAOptimizer
from torchopt.optim.objective import Accuracy, Macs, ObjectiveContainer
from torchopt.optim.constraint import ChannelConstraint

from thop import profile


# Get your trained model
model = torch.load('path/to/trained/model.pth')

# Get dataset, on which model was trained. Dataset should be divided to training, validation 
# and testing set. Validation set will be used for measuring accuracy of pruned model.
train_set, val_set, test_set = get_dataset()

# Define model's input shape
input_shape = (1, 3, 32, 32)

# Specify device, on wich optimization will be performed
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prunable modules are all linear and convolutional layers in the net
names = [name for name, _ in utils.prunable_modules(model)]
bounds = [(0, len(module.weight) - 1) for _, module in utils.prunable_modules(model)]
pruner = ChannelPruner(names, input_shape)

# Create GA optimizer
optimizer = IntegerGAOptimizer(
    ind_size=len(names),
    pop_size=100,
    elite_num=10,
    tourn_size=10,
    n_gen=30,
    mutp=0.1,
    mut_indp=0.05,
    cx_indp=0.5,
    bounds=bounds
)

sample = torch.randn(input_shape, device=device)
orig_acc = utils.evaluate(model, test_data, device)
orig_macs, _ = profile(model, inputs=(sample,), verbose=False)

# Create composed objective function to get best trade-off between model accuracy and MACs reduction
acc = Accuracy(model=model, pruner=pruner, weight=1.0, val_data=val_set, orig_acc=orig_acc)
macs = Macs(model=model, pruner=pruner, orig_macs=orig_macs, weight=1.0, in_shape=input_shape)
objective = ObjectiveContainer(acc, macs)
constraint = ChannelConstraint(model=model, pruner=pruner)

# Perform optimization
solution = optimizer.maximize(objective, constraint)

# Get pruned model according to best solution found by GA
pruned_model = pruner.prune(model=model, mask=solution)

```

## License

[MIT](https://choosealicense.com/licenses/mit/)