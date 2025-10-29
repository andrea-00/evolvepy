# src/evolvepy/individual.py

from typing import TypeVar, Generic

# Define a generic TypeVariable for the genotype.
GenotypeType = TypeVar('GenotypeType')


class Individual(Generic[GenotypeType]):
    """
    Represents a single candidate solution within the EA population.

    This class acts as a generic container for the "genetic material" 
    (the genotype) and its corresponding evaluated score (the fitness).
    It is designed to be agnostic to the specific problem, allowing
    the genotype to be any data structure (e.g., list, numpy array,
    custom tree structure).
    """

    def __init__(self, genotype: GenotypeType):
        """
        Initializes a new Individual with its genetic material.

        The fitness is explicitly set to None, indicating that the
        individual has not yet been evaluated by the fitness function.

        Args:
            genotype (GenotypeType): The data structure representing a
                potential solution in the search space.
        """
        # The core solution data. This is what the genetic
        # operators (mutation, crossover) will act upon.
        self.genotype: GenotypeType = genotype
        
        # The scalar score of the individual's performance.
        # It is 'None' until the evaluate() method is called on it.
        self.fitness: float | None = None

        # The age of an individual
        self.age: int = 0

    def __repr__(self) -> str:
        """
        Returns an unambiguous, official string representation of the Individual.

        This representation is primarily intended for debugging and logging.
        It clearly shows the individual's fitness, which is the key
        metric for its performance.

        Returns:
            str: A string representation of the individual, e.g.,
                 "Individual(fitness=123.45)" or "Individual(fitness=N/A)".
        """
        # Format fitness to 4 decimal places for readability,
        # or show 'N/A' if the individual is unevaluated.
        fit_str = f"{self.fitness:.4f}" if self.fitness is not None else "N/A"
        
        return f"Individual(fitness={fit_str}, age={self.age})"