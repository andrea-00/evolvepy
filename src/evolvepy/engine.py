# src/evolvepy/engine.py

from typing import Callable, List, Generic, Dict, Any
import numpy as np
import math

from .individual import Individual, GenotypeType
from .logger import BaseLogger, NullLogger


class EvolutionaryAlgorithm(Generic[GenotypeType]):
    """
    A generic, strategy-based evolutionary algorithm engine.

    This class provides the core evolutionary loop (initialization,
    evaluation, selection, reproduction). It is designed to be
    problem-agnostic. The specific behaviors are "injected" as 
    strategy functions (callables) during initialization.

    This engine assumes a maximization problem (higher fitness is better).
    """

    def __init__(
        self,
        # Problem-Specific Callables
        fitness_function: Callable[[GenotypeType], float],
        initialization_function: Callable[[], GenotypeType],

        # Core Strategy Callables
        parent_selection: Callable[[List[Individual[GenotypeType]]], List[Individual[GenotypeType]]],
        reproduction_strategy: Callable[
            [List[Individual[GenotypeType]], int],  # [parents, current_gen]
            List[Individual[GenotypeType]]           # returns offspring
        ],
        survivor_selection: Callable[[List[Individual[GenotypeType]], List[Individual[GenotypeType]]], List[Individual[GenotypeType]]],
        
        # Engine Parameters
        population_size: int,

        # Services
        logger: BaseLogger | None = None
    ):
        """
        Initializes the Evolutionary Algorithm engine.

        Args:
            fitness_function: A callable that takes one 'genotype' and
                returns a float representing its fitness score.
            
            initialization_function: A callable that takes no arguments
                and returns a single, new 'genotype'.
            
            parent_selection: A callable that takes the current population
                (List[Individual]) and returns a list of Individuals selected
                to be parents.
            
            reproduction_strategy: A callable that takes the list of parents
                and the current generation number, and returns a new list of 
                offspring (Individuals). This strategy encapsulates all logic
                for creating new individuals (e.g., crossover, mutation).
            
            survivor_selection: A callable that takes two lists 
                (the old population, the new offspring) and returns a single
                list representing the new population for the next generation.
            
            population_size: The target number of individuals to maintain
                in the population at each generation.

            logger: An object that implements the BaseLogger interface.
                    If None, a NullLogger (which does nothing) is used.
        """
        
        # Store problem-specific functions
        self.evaluate_genotype = fitness_function
        self.create_genotype = initialization_function
        
        # Store strategy functions
        self.select_parents = parent_selection
        self.reproduce = reproduction_strategy
        self.select_survivors = survivor_selection
        
        # Store engine parameters
        self.population_size = population_size
        
        # Internal State
        self.population: List[Individual[GenotypeType]] = []
        self.current_generation: int = 0
        self.best_individual: Individual[GenotypeType] | None = None

        # Services
        self.logger: BaseLogger = logger if logger is not None else NullLogger()

        # Statistics History
        self.history: Dict[str, List] = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "std_fitness": [],
            "worst_fitness": []
        }
    
    def _get_config(self) -> Dict[str, Any]:
        """Private helper to gather engine configuration for logging."""
        return {
            "Algorithm": self.__class__.__name__,
            "Population Size": self.population_size,
            "Parent Selection": self.select_parents.__class__.__name__,
            "Reproduction": self.reproduce.__class__.__name__,
            "Survivor Selection": self.select_survivors.__class__.__name__,
        }

    def _evaluate_population(self, population: List[Individual[GenotypeType]]):
        """
        Evaluates all individuals in a given list that do not have a fitness score.

        This method calls the user-provided 'evaluate_genotype' function
        and updates the 'fitness' attribute of each individual.

        It also updates the 'best_individual' tracker if a new
        best is found.
        """
        for ind in population:
            if ind.fitness is None:
                ind.fitness = self.evaluate_genotype(ind.genotype)
                
                if (self.best_individual is None or 
                    ind.fitness > self.best_individual.fitness):
                    self.best_individual = ind
    
    def _update_history_and_log(self, generation: int):
        """
        Private helper to calculate stats, update history, and log.
        """
        if not self.population:
            # Avoid errors if population is empty
            return

        # Calculate stats from the current population
        current_fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        if not current_fitnesses:
             # Avoid errors if no individuals have fitness (e.g., failed eval)
            return

        best_gen_fit = np.max(current_fitnesses)
        mean_gen_fit = np.mean(current_fitnesses)
        std_gen_fit = np.std(current_fitnesses)
        worst_gen_fit = np.min(current_fitnesses)
        
        # Update history
        self.history["generation"].append(generation)
        self.history["best_fitness"].append(best_gen_fit)
        self.history["mean_fitness"].append(mean_gen_fit)
        self.history["std_fitness"].append(std_gen_fit)
        self.history["worst_fitness"].append(worst_gen_fit)
        
        # Find the best individual *in this generation*
        # NOTE: self.best_individual tracks the *all-time* best
        best_in_gen = max(self.population, key=lambda i: i.fitness)
        
        # Call the logger
        self.logger.log_generation(generation, self.history, best_in_gen)

    def run(self, generations: int) -> tuple[Individual[GenotypeType], Dict[str, List]]:
        """
        Executes the main evolutionary loop for a specified number of generations.

        Args:
            generations: The total number of generations to run the evolution for.

        Returns:
            a tuple containing:
                Individual: The best individual found across all generations.
                Dict[str, List]: The history of the EA
        
        Raises:
            ValueError: If the initial population is empty or an error
                        occurs during evaluation.
        """
        
        # Inizialization
        self.logger.log_start(self._get_config())
        initial_genotypes = [self.create_genotype() for _ in range(self.population_size)]
        self.population = [Individual(gen) for gen in initial_genotypes]
        self._evaluate_population(self.population)

        if self.best_individual is None:
            raise ValueError("Initial population evaluation failed to produce a valid individual.")

        self._update_history_and_log(generation=0)

        # Evolutionary Loop
        for gen in range(1, generations + 1):
            self.current_generation = gen
            
            # Parent Selection
            parents = self.select_parents(self.population)
            
            # Reproduction
            # This single call handles all offspring creation logic.
            # We pass the current generation to allow for parameter scheduling.
            offspring = self.reproduce(parents, self.current_generation)
            
            # Offspring Evaluation
            self._evaluate_population(offspring)

            # Increment the age of all individuals in the *current*
            # population before survivor selection.
            # The 'offspring' list correctly has age=0.
            for ind in self.population:
                ind.age += 1
            
            # Survivor Selection
            self.population = self.select_survivors(self.population, offspring)

            # Stats Collection & Logging
            self._update_history_and_log(generation=gen)

        self.logger.log_end(self.best_individual, generations)
        self.logger.close()
        return self.best_individual, self.history