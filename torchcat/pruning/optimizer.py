import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable, List, Tuple

import numpy as np
from deap import creator, tools
from deap.base import Toolbox, Fitness

from .constraint import Constraint
from .objective import Objective


class Optimizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def minimize(self, objective: Objective, constraint: Constraint) -> Any:
        pass

    @abstractmethod
    def maximize(self, objective: Objective, constraint: Constraint) -> Any:
        pass


class GAOptimizer(Optimizer):
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
        init_pop: Iterable[Any] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self._ind_size = ind_size
        self._pop_size = pop_size
        self._elite_num = elite_num
        self._tourn_size = tourn_size
        self._n_gen = n_gen
        self._mutp = mutp
        self._mut_indp = mut_indp
        self._cx_indp = cx_indp
        self._init_pop = init_pop
        self._verbose = verbose

        self._best = None
        self._population = None
        self._toolbox = None
        self._history = None

    def minimize(self, objective: Objective, constraint: Constraint) -> Any:
        creator.create("FitnessMin", Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self._optimize(objective, constraint)

    def maximize(self, objective: Objective, constraint: Constraint) -> Any:
        creator.create("FitnessMax", Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self._optimize(objective, constraint)

    def best(self) -> Any:
        return self._best

    def population(self) -> Iterable[Any]:
        return self._population

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

        for gen in range(1, self._n_gen + 1):
            new_pop = list(map(self._toolbox.clone, self._elite_set(self._population)))

            while len(new_pop) < len(self._population):
                off1, off2 = self._crossover(self._population)
                off1, off2 = self._mutation(off1), self._mutation(off2)

                if constraint.feasible(off1):
                    new_pop.append(off1)

                if constraint.feasible(off2):
                    new_pop.append(off2)

            self._population = new_pop
            self._handle_generation(gen, objective)

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

    def _init_population(self, ind_cls: Any) -> Iterable[Any]:
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
        init_pop: Iterable[Any] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            ind_size,
            pop_size,
            elite_num,
            tourn_size,
            n_gen,
            mutp,
            mut_indp,
            cx_indp,
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

                if constraint.feasible(ind):
                    pop.append(ind)

                if len(pop) == self._pop_size:
                    break

        return pop


class IntegerGAOptimizer(GAOptimizer):
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
        init_pop: Iterable[Any] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            ind_size,
            pop_size,
            elite_num,
            tourn_size,
            n_gen,
            mutp,
            mut_indp,
            cx_indp,
            init_pop,
            verbose,
        )

        self._bounds = bounds

    def _define_operations(self) -> Toolbox:
        tb = Toolbox()

        lbounds = [t[0] for t in self._bounds]
        ubounds = [t[1] for t in self._bounds]

        tb.register("mate", tools.cxUniform, indpb=self._cx_indp)
        tb.register("mutate", tools.mutUniformInt, low=lbounds, up=ubounds, indpb=self._mut_indp)
        tb.register("select", tools.selTournament, tournsize=self._tourn_size)

        return tb

    def _generate_population(self, ind_cls: type, constraint: Constraint) -> Iterable[Any]:
        pop = []

        while len(pop) < self._pop_size:
            ind = ind_cls([random.randint(low, up) for low, up in self._bounds])

            if constraint.feasible(ind):
                pop.append(ind)

        return pop
