import random
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, List, Tuple

import numpy as np
from deap import creator, tools
from deap.base import Fitness, Toolbox

from . import utils
from .constraint import Constraint
from .objective import Objective

warnings.simplefilter("ignore", RuntimeWarning)


class Optimizer(ABC):
    """Abstract class representing optimization algortihm."""

    def __init__(self) -> None:
        """Ctor."""
        pass

    @abstractmethod
    def minimize(self, objective: Objective, constraint: Constraint) -> Any:
        """Function that minimizes objective function given as a parameter subject to constraints
        given as a parameter, i. e.:

        min : objective
        s.t.: constraint

        Args:
            objective (Objective): Objective function to be minimized.
            constraint (Constraint): Constraints of the optimization problem.

        Returns:
            Any: Best found solution.
        """
        pass

    @abstractmethod
    def maximize(self, objective: Objective, constraint: Constraint) -> Any:
        """Function that maximizes objective function given as a parameter subject to constraints
        given as a parameter, i. e.:

        max : objective
        s.t.: constraint

        Args:
            objective (Objective): Objective function to be maximized.
            constraint (Constraint): Constraints of the optimization problem.

        Returns:
            Any: Best found solution.
        """
        pass


class GAOptimizer(Optimizer):
    """Represents base class for genetic algorithm (GA).

    Genetic algorithm (GA) implementation used to (but not limited to) solving neural 
    network pruning problem. Neural network pruning can be formulated as optimization 
    problem to find best subset from the set of network's filters/neurons, i. e.:

    max : accuracy(model(W • M))
    s.t.: resource(model(W • M)) <= budged

    where W are network's filters/neurons, M is mask produced by optimization, resource 
    can be any resource we want to reduce (e. g. MACs, latency, model size, ...) and 
    budget is our desired upper bound of the resource we want to reduce.

    This GA implementation uses basic principles and techniques to perform evolutionary
    process. Tournament selection is used to select two individuals for the crossover
    operation. Elite set is kept to ensure quality of the new population. Crossover and
    mutation operations are defined in specific implementations of this abstract class.
    """

    def __init__(
        self,
        ind_size: int,
        pop_size: int,
        elite_num: int,
        tourn_size: int,
        n_gen: int,
        mutp: float,
        mut_indp: float,
        cx_indp: float,
        early_stop: int = -1,
        init_pop: Iterable[Any] = None,
        verbose: bool = True,
    ) -> None:
        """Ctor.

        Args:
            ind_size (int): Size of the individual.
            pop_size (int): Size of the population.
            elite_num (int): Elite set size in range <0, pop_size).
            tourn_size (int): Tournament size in range <2, pop_size).
            n_gen (int): Number of produced generations.
            mutp (float): Mutation probablity of new individual.
            mut_indp (float): Probability of mutating single bit of new individual.
            cx_indp (float): Probability used in uniform crossover.
            early_stop (int, optional): Early stopping. Defaults to -1.
            init_pop (Iterable[Any], optional): Initial population. Defaults to None.
            verbose (bool, optional): Verbosity of the output. Defaults to True.
        """
        super().__init__()

        self._ind_size = ind_size
        self._pop_size = pop_size
        self._elite_num = elite_num
        self._tourn_size = tourn_size
        self._n_gen = n_gen
        self._mutp = mutp
        self._mut_indp = mut_indp
        self._cx_indp = cx_indp
        self._early_stop = early_stop if early_stop > 0 else None
        self._init_pop = init_pop
        self._verbose = verbose

        self._best = None
        self._population = None
        self._toolbox = None
        self._history = None

    def minimize(self, objective: Objective, constraint: Constraint) -> Any:
        creator.create("FitnessMin", Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        return self._optimize(objective, constraint)

    def maximize(self, objective: Objective, constraint: Constraint) -> Any:
        creator.create("FitnessMax", Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        return self._optimize(objective, constraint)

    def history(self) -> Any:
        return self._history

    def _optimize(self, objective: Objective, constraint: Constraint) -> Any:
        self._best = None
        self._toolbox = self._define_operations()
        self._history = tools.Logbook()

        self._population = (
            self._generate_population(creator.Individual, constraint)
            if self._init_pop is None
            else self._init_population(creator.Individual)
        )
        self._handle_generation(0, objective)

        curr_max = self._best.fitness
        no_improve = 0

        for gen in range(1, self._n_gen + 1):
            new_pop = list(map(self._toolbox.clone, self._elite_set(self._population)))

            while len(new_pop) < len(self._population):
                off1, off2 = self._crossover(self._population)
                off1, off2 = self._mutation(off1), self._mutation(off2)

                if (constraint is None or constraint.feasible(off1)) and off1 not in new_pop:
                    new_pop.append(off1)

                if (constraint is None or constraint.feasible(off2)) and off2 not in new_pop:
                    new_pop.append(off2)

            self._population = new_pop
            self._handle_generation(gen, objective)

            # Check if current best solution was changed
            if self._best.fitness > curr_max:
                curr_max = self._best.fitness
                no_improve = 0
            else:
                no_improve += 1

            # No improvement has been made, early stopping the optimization
            if self._early_stop is not None and no_improve == self._early_stop:
                time = datetime.now().strftime("%H:%M:%S")
                print(
                    f"{time} - No improvement has been made in {no_improve} generations, early stopping"
                )
                break

        return self._best

    def _handle_generation(self, gen_num: int, objective: Objective) -> None:
        # Evaluate population
        for individual in self._population:
            if not individual.fitness.values:
                individual.fitness.values = objective.evaluate(individual)

        # Keep current best found solution
        self._best = (
            self._keep_best(self._best, self._population)
            if self._best is not None
            else tools.selBest(self._population, k=1)[0]
        )

        # Save statistics and state of current population
        stats = self._create_stats()
        record = stats.compile(self._population)
        self._history.record(gen=gen_num, **record, best=self._best, pop=self._population)

        # Print statistcs to terminal
        if self._verbose:
            stats_str = ", ".join([f"{k.capitalize()} = {v:.4f}" for k, v in record.items()])
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} - Generation {gen_num:04d}: {stats_str}")

    def _init_population(self, ind_cls: type) -> Iterable[Any]:
        return [ind_cls(content) for content in self._init_pop]

    def _crossover(self, population: Iterable[Any]) -> Tuple[Any]:
        # Parent selection
        p1, p2 = self._toolbox.select(population, 2)
        off1, off2 = self._toolbox.clone(p1), self._toolbox.clone(p2)

        # Crossover
        off1, off2 = self._toolbox.mate(off1, off2)
        del off1.fitness.values
        del off2.fitness.values

        return (off1, off2)

    def _mutation(self, individual: Any) -> Any:
        if random.random() <= self._mutp:
            individual = self._toolbox.mutate(individual)[0]
            del individual.fitness.values

        return individual

    def _elite_set(self, population: Iterable[Any]) -> List[Any]:
        return tools.selBest(population, k=self._elite_num)

    def _keep_best(self, curr_best: Any, population: Iterable[Any]) -> Any:
        return tools.selBest([curr_best] + tools.selBest(population, k=1), k=1)[0]

    def _create_stats(self) -> tools.Statistics:
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)

        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        return stats

    @abstractmethod
    def _define_operations(self) -> Toolbox:
        pass

    @abstractmethod
    def _generate_population(self, ind_cls: type, constraint: Constraint) -> Iterable[Any]:
        pass


class BinaryGAOptimizer(GAOptimizer):
    """GA implementation modelling binary optimization problem.

    Genetic algorithm (GA) implementation used to (but not limited to) solving neural 
    network pruning problem defined as binary optimization problem. Number of decision
    variables in this problem is equal to number of filters/neurons in the network. 
    
    If value of the decision variable xi = 0, i-th filter/neuron will be pruned, if xi = 1, 
    i-th filter/neuron will be preserved in the network.

    In this implementation, crossover operation is performed in uniform fashion (uniform 
    crossover), mutation operation is performed by randomly flipping the elements of the
    individual, i. e.: xi = !xi_actual.
    """

    def __init__(
        self,
        ind_size: int,
        pop_size: int,
        elite_num: int,
        tourn_size: int,
        n_gen: int,
        mutp: float,
        mut_indp: float,
        cx_indp: float,
        early_stop: int = -1,
        init_pop: Iterable[Any] = None,
        verbose: bool = True,
    ) -> None:
        """Ctor.

        Args:
            ind_size (int): Size of the individual.
            pop_size (int): Size of the population.
            elite_num (int): Elite set size in range <0, pop_size).
            tourn_size (int): Tournament size in range <2, pop_size).
            n_gen (int): Number of produced generations.
            mutp (float): Mutation probablity of new individual.
            mut_indp (float): Probability of mutating single bit of new individual.
            cx_indp (float): Probability used in uniform crossover.
            early_stop (int, optional): Early stopping. Defaults to -1.
            init_pop (Iterable[Any], optional): Initial population. Defaults to None.
            verbose (bool, optional): Verbosity of the output. Defaults to True.
        """
        super().__init__(
            ind_size,
            pop_size,
            elite_num,
            tourn_size,
            n_gen,
            mutp,
            mut_indp,
            cx_indp,
            early_stop,
            init_pop,
            verbose,
        )

    def _define_operations(self) -> Toolbox:
        tb = Toolbox()

        tb.register("mate", tools.cxUniform, indpb=self._cx_indp)
        tb.register("mutate", tools.mutFlipBit, indpb=self._mut_indp)
        tb.register("select", tools.selTournament, tournsize=self._tourn_size)

        return tb

    def _generate_population(self, ind_cls: type, constraint: Constraint) -> Iterable[Any]:
        pop = []

        while len(pop) < self._pop_size:
            for i in range(self._pop_size):
                p = (i + 1) / self._pop_size
                ind = ind_cls([random.random() <= p for _ in range(self._ind_size)])

                if (constraint is None or constraint.feasible(ind)) and ind not in pop:
                    pop.append(ind)

                if len(pop) == self._pop_size:
                    break

        return pop


class IntegerGAOptimizer(GAOptimizer):
    """GA implementation modelling integer optimization problem.

    Genetic algorithm (GA) implementation used to (but not limited to) solving neural network 
    pruning problem defined as integer optimization problem. Number of decision variables in 
    this problem is equal to number layers in the network that will be pruned. 
    
    Value of the decision variable xi defines number of filters/neurons that will be pruned
    in the i-th layer of the neural network.

    In this implementation, crossover operation is performed in uniform fashion (uniform 
    crossover), mutation operation is performed by generating random number from triangular
    distribution with min/max defined by lower/upper bounds (minimum and maximum of pruned
    filters/neurons in specific layer) and mode is defined by current value of the element,
    i. e.: xi ~ triangular(min, xi_actual, max).
    """

    def __init__(
        self,
        ind_size: int,
        pop_size: int,
        elite_num: int,
        tourn_size: int,
        n_gen: int,
        mutp: float,
        mut_indp: float,
        cx_indp: float,
        bounds: Iterable[Tuple[int, int]],
        early_stop: int = -1,
        init_pop: Iterable[Any] = None,
        verbose: bool = True,
    ) -> None:
        """Ctor.

        Args:
            ind_size (int): Size of the individual.
            pop_size (int): Size of the population.
            elite_num (int): Elite set size in range <0, pop_size).
            tourn_size (int): Tournament size in range <2, pop_size).
            n_gen (int): Number of produced generations.
            mutp (float): Mutation probablity of new individual.
            mut_indp (float): Probability of mutating single bit of new individual.
            cx_indp (float): Probability used in uniform crossover.
            bounds (Iterable[Tuple[int, int]]): Pairs of maximum/minimum number of filters/neurons
                to be pruned in i-th layer of the network.
            early_stop (int, optional): Early stopping. Defaults to -1.
            init_pop (Iterable[Any], optional): Initial population. Defaults to None.
            verbose (bool, optional): Verbosity of the output. Defaults to True.
        """
        super().__init__(
            ind_size,
            pop_size,
            elite_num,
            tourn_size,
            n_gen,
            mutp,
            mut_indp,
            cx_indp,
            early_stop,
            init_pop,
            verbose,
        )

        self._bounds = bounds

    def _define_operations(self) -> Toolbox:
        tb = Toolbox()

        lbounds = [t[0] for t in self._bounds]
        ubounds = [t[1] for t in self._bounds]

        tb.register("mate", tools.cxUniform, indpb=self._cx_indp)
        tb.register("mutate", utils.mut_triangular, low=lbounds, up=ubounds, indpb=self._mut_indp)
        tb.register("select", tools.selTournament, tournsize=self._tourn_size)

        return tb

    def _generate_population(self, ind_cls: type, constraint: Constraint) -> Iterable[Any]:
        pop = []

        while len(pop) < self._pop_size:
            for i in range(self._pop_size + 1):
                ind_content = []
                p = i / self._pop_size

                for low, up in self._bounds:
                    size = up - low
                    mode = p * (up - low) + low
                    left = max(low, mode - 0.1 * size)
                    right = min(up, mode + 0.1 * size)

                    ind_content.append(int(random.triangular(low=left, high=right, mode=mode)))

                ind = ind_cls(ind_content)

                if (constraint is None or constraint.feasible(ind)) and ind not in pop:
                    pop.append(ind)

                if len(pop) == self._pop_size:
                    break

        return pop
