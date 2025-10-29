# src/evolvepy/base_problem.py

from abc import ABC, abstractmethod
from typing import Callable, Any, TypeVar, Generic

# Define a generic type for the genotype
GenotypeType = TypeVar('GenotypeType')

class BaseProblem(ABC, Generic[GenotypeType]):
    """
    Abstract Base Class (ABC) for a problem definition.

    This class defines the "contract" that any problem must fulfill
    to be solvable by the EvolutionaryAlgorithm engine.
    
    It ensures that every problem provides the two essential,
    problem-specific callables:
    1. A function to create a new, random genotype.
    2. A function to evaluate the fitness of a given genotype.
    """

    @abstractmethod
    def get_fitness_function(self) -> Callable[[GenotypeType], float]:
        """
        Must return the problem-specific fitness function.

        The returned callable must take one argument (the genotype)
        and return a single float (the fitness).
        
        Returns:
            A callable (e.g., self._calculate_fitness)
        """
        pass

    @abstractmethod
    def get_initializer_function(self) -> Callable[[], GenotypeType]:
        """
        Must return the problem-specific genotype initializer function.

        The returned callable must take no arguments and return
        a single, new genotype.
        
        Returns:
            A callable (e.g., self._create_genotype)
        """
        pass