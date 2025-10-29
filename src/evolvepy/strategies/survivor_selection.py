# src/evolvepy/strategies/survivor_selection.py

from typing import List

from ..individual import Individual, GenotypeType

class PlusSelection:
    """
    Implements a (μ + λ) survivor selection strategy.

    This strategy combines the old population (μ) and the new offspring (λ)
    into a single pool. It then selects the 'population_size' best
    individuals from this combined pool to form the next generation.
    
    This guarantees that the best individual found is never lost
    (monotonic improvement).
    """
    
    def __init__(self, population_size: int):
        """
        Args:
            population_size (int): The target size of the population to
                                   maintain (the 'μ' value).
        """
        self.population_size = population_size

    def __call__(self, 
                 old_population: List[Individual[GenotypeType]], 
                 offspring: List[Individual[GenotypeType]]) -> List[Individual[GenotypeType]]:
        """
        Args:
            old_population: The list of individuals from the
                            previous generation (μ).
            offspring: The list of new individuals generated (λ).

        Returns:
            The next generation (List[Individual] of size 'population_size').
        """
        # Combine pools
        combined_population = old_population + offspring
        
        # Sort by fitness (descending)
        combined_population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Select the top 'μ' individuals
        return combined_population[:self.population_size]



class CommaSelection:
    """
    Implements a (μ, λ) survivor selection strategy.

    This strategy selects the next generation *only* from the
    newly created offspring (λ). The entire old population (μ) is
    discarded, regardless of fitness.
    
    This strategy requires that λ >= μ.
    
    It is effective at escaping local optima, as it
    "forgets" old solutions, but it does not guarantee
    monotonic improvement (the best solution can be lost).
    """
    
    def __init__(self, population_size: int):
        """
        Args:
            population_size (int): The target size of the population to
                                   maintain (the 'μ' value).
        """
        self.population_size = population_size

    def __call__(self, 
                 old_population: List[Individual[GenotypeType]], 
                 offspring: List[Individual[GenotypeType]]) -> List[Individual[GenotypeType]]:
        """
        Args:
            old_population: The old population (μ). This is *ignored*.
            offspring: The new offspring (λ). This is the *only* pool
                       from which survivors are chosen.

        Returns:
            The next generation (List[Individual] of size 'population_size').
        """
        # Check if we have enough offspring
        if len(offspring) < self.population_size:
            raise ValueError(f"(μ, λ) selection requires λ ({len(offspring)}) "
                             f"to be >= μ ({self.population_size})")

        # Sort *only the offspring* by fitness (descending)
        offspring.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Select the top 'μ' individuals from the offspring
        return offspring[:self.population_size]



class PlusAgeBasedSelection:
    """
    Implements a (μ + λ) strategy that also considers age.
    
    This strategy combines parents (μ) and offspring (λ), but
    it discards any individual (even a high-fitness one) whose
    age exceeds 'max_age' *before* performing the final elitist selection.
    
    This helps combat premature convergence by "retiring"
    old individuals, even if they are local optima.
    """
    
    def __init__(self, population_size: int, max_age: int):
        """
        Args:
            population_size (int): The target size of the population (μ).
            max_age (int): The maximum number of generations an
                             individual can survive.
        """
        self.population_size = population_size
        self.max_age = max_age

    def __call__(self, 
                 old_population: List[Individual[GenotypeType]], 
                 offspring: List[Individual[GenotypeType]]) -> List[Individual[GenotypeType]]:
        """
        Args:
            old_population: The list of individuals from the
                            previous generation (μ).
            offspring: The list of new individuals generated (λ).

        Returns:
            The next generation (List[Individual] of size 'population_size').
        """
        # Combine pools
        combined_population = old_population + offspring
        
        # Filter: Keep only individuals within the age limit
        # (Offspring always pass, as their age is 0)
        eligible_population = [
            ind for ind in combined_population 
            if ind.age <= self.max_age
        ]
        
        # Sort the *eligible* population by fitness
        eligible_population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Select the top 'μ', but handle the case where
        # filtering left us with fewer than 'μ' individuals.
        survivors = eligible_population[:self.population_size]
        
        # TODO:
        # Re-fill if filtering was too aggressive
        # if len(survivors) < self.population_size:
        #    ... (add logic to fill with best-of-the-rest, etc.)
        
        return survivors