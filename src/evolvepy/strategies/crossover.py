# src/evolvepy/strategies/crossover.py

import random
from typing import List, Union, Callable, Any

from ..individual import Individual, GenotypeType


class OrderedCrossover:
    """
    Performs Ordered Crossover (OX1) on two parent genotypes.

    This crossover operator is designed for **permutation-based
    representations**. It selects a random subsequence (swath)
    from the first parent, copies it directly to the child,
    and then fills the remaining positions with genes from the
    second parent in the order they appear, skipping any genes
    already present from the first parent's swath.

    This guarantees the child is a valid permutation.
    """

    def __init__(self, 
                 crossover_rate: Union[float, Callable[[int], float]] = 0.9):
        """
        Initializes the ordered crossover strategy.

        Args:
            crossover_rate: The probability (0.0 to 1.0) that
                two selected parents will undergo crossover.
                If crossover does not occur, the offspring are
                direct clones of the parents.
                This can be a fixed float or a callable scheduler (fn(gen) -> float).
        """
        self.rate_schedule = crossover_rate
    
    def _create_child(self, 
                      parent1_gen: List[GenotypeType], 
                      parent2_gen: List[GenotypeType], 
                      start: int, 
                      end: int) -> List[GenotypeType]:
        """
        Creates a single child using the OX1 (Ordered Crossover) logic.

        It copies a swath from parent1 (from 'start' to 'end') and
        then fills the remaining slots with genes from parent2
        in their original, circular order.

        Args:
            parent1_gen: The genotype (list) of the first parent.
            parent2_gen: The genotype (list) of the second parent.
            start: The start index (inclusive) of the swath from parent1.
            end: The end index (inclusive) of the swath from parent1.

        Returns:
            A new child genotype (list).
        """
        gen_len = len(parent1_gen)
        child_gen = [None] * gen_len
        
        # Copy the swath from parent1
        child_gen[start:end+1] = parent1_gen[start:end+1]
        
        genes_in_child = set(child_gen[start:end+1])
        
        # Fill remaining slots from parent2, maintaining order
        
        # Pointer to read from parent2
        p2_read_idx = (end + 1) % gen_len
        
        # Pointer to write into the child
        child_write_idx = (end + 1) % gen_len
        
        # Loop until the child is full (no more 'None' values)
        # We only need to iterate gen_len times at most
        for _ in range(gen_len):
            # Find the next gene in parent2
            gene = parent2_gen[p2_read_idx]
            
            # Check if this gene is already in the child
            if gene not in genes_in_child:
                # If not, write it to the child's current write position
                child_gen[child_write_idx] = gene
                
                # Advance the *write* pointer (circularly)
                child_write_idx = (child_write_idx + 1) % gen_len
            
            # Always advance the *read* pointer (circularly)
            p2_read_idx = (p2_read_idx + 1) % gen_len
            
            # Optimization: if write pointer has caught up to start, we are done
            if child_write_idx == start:
                break
                
        return child_gen

    def __call__(self, 
                 parents: List[Individual[List[GenotypeType]]], 
                 current_generation: int) -> List[Individual[List[GenotypeType]]]:
        """
        Creates a new list of offspring from a list of parents.

        Args:
            parents: The list of parent Individuals selected for reproduction.
            current_generation: The current generation number (for scheduling).

        Returns:
            A new list of offspring Individuals. Fitness is set to None.
            
        Raises:
            TypeError: If a genotype is not a 'list'.
            ValueError: If parent genotypes have different lengths.
        """
        
        # Resolve the probability for this generation
        rate = self.rate_schedule(current_generation) if callable(self.rate_schedule) else self.rate_schedule
            
        offspring = []
        
        # Process parents in pairs
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i+1] if (i + 1) < len(parents) else None

            if p2 is None:
                # Handle odd number of parents: clone the last one
                offspring.append(Individual(p1.genotype[:]))
                continue

            # Check types and lengths
            if not (isinstance(p1.genotype, list) and isinstance(p2.genotype, list)):
                raise TypeError(f"OrderedCrossover requires 'list' genotypes, "
                                f"got {type(p1.genotype)} and {type(p2.genotype)}")
            
            gen_len = len(p1.genotype)
            if gen_len != len(p2.genotype):
                 raise ValueError("Genotypes must have the same length for crossover.")
            if gen_len < 2:
                 # Crossover is not meaningful
                 offspring.append(Individual(p1.genotype[:]))
                 offspring.append(Individual(p2.genotype[:]))
                 continue

            # Check if crossover should occur
            if random.random() > rate:
                # No crossover, offspring are clones
                offspring.append(Individual(p1.genotype[:]))
                offspring.append(Individual(p2.genotype[:]))
                continue

            # Perform Crossover
            # Select two distinct cut points
            cut1, cut2 = random.sample(range(gen_len), 2)
            start, end = min(cut1, cut2), max(cut1, cut2)

            # 3. Create the two children
            child1 = self._create_child(p1.genotype, p2.genotype, start, end)
            child2 = self._create_child(p2.genotype, p1.genotype, start, end)

            # 4. Add new Individuals to the offspring list
            offspring.append(Individual(child1))
            offspring.append(Individual(child2))
            
        return offspring


class CycleCrossover:
    """
    Performs Cycle Crossover (CX) on two parent genotypes.

    This operator is also designed for **permutation-based
    representations**. It identifies "cycles" of gene positions
    between the two parents. For each cycle, the genes are
    copied from one parent to the corresponding positions in the
    child, ensuring the child inherits a valid permutation.
    
    

    It is highly conservative, preserving the absolute positions
    of many genes from the parents.
    """
    
    def __init__(self, 
                 crossover_rate: Union[float, Callable[[int], float]] = 0.9):
        """
        Initializes the cycle crossover strategy.

        Args:
            crossover_rate: The probability (0.0 to 1.0) that
                two selected parents will undergo crossover.
                This can be a fixed float or a callable scheduler (fn(gen) -> float).
        """
        self.rate_schedule = crossover_rate
        
    def __call__(self, 
                 parents: List[Individual[List[GenotypeType]]], 
                 current_generation: int) -> List[Individual[List[GenotypeType]]]:
        """
        Creates a new list of offspring from a list of parents.

        Args:
            parents: The list of parent Individuals selected for reproduction.
            current_generation: The current generation number (for scheduling).

        Returns:
            A new list of offspring Individuals.
        """
        
        rate = self.rate_schedule(current_generation) if callable(self.rate_schedule) else self.rate_schedule
            
        offspring = []
        
        # Process parents in pairs
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i+1] if (i + 1) < len(parents) else None

            if p2 is None:
                offspring.append(Individual(p1.genotype[:]))
                continue

            if not (isinstance(p1.genotype, list) and isinstance(p2.genotype, list)):
                raise TypeError(f"CycleCrossover requires 'list' genotypes.")
            
            gen_len = len(p1.genotype)
            if gen_len != len(p2.genotype):
                 raise ValueError("Genotypes must have the same length for crossover.")

            if random.random() > rate:
                offspring.append(Individual(p1.genotype[:]))
                offspring.append(Individual(p2.genotype[:]))
                continue

            # --- Perform Crossover ---
            
            # Helper to find the index of a value in a list (genotype)
            # (Creating a lookup map is faster for large genotypes)
            p2_value_to_index = {value: idx for idx, value in enumerate(p2.genotype)}

            # Initialize children as None-filled lists
            c1_gen = [None] * gen_len
            c2_gen = [None] * gen_len
            
            # Keep track of which indices we've already visited
            visited_indices = [False] * gen_len
            
            cycle_num = 0
            
            for idx in range(gen_len):
                if visited_indices[idx]:
                    continue # This index is already part of a cycle
                    
                cycle_num += 1
                current_idx = idx
                
                # Find all indices in the current cycle
                while not visited_indices[current_idx]:
                    visited_indices[current_idx] = True
                    
                    # Alternate which parent to copy from based on cycle number
                    if cycle_num % 2 == 1:
                        # Odd cycle: c1 gets from p1, c2 gets from p2
                        c1_gen[current_idx] = p1.genotype[current_idx]
                        c2_gen[current_idx] = p2.genotype[current_idx]
                    else:
                        # Even cycle: c1 gets from p2, c2 gets from p1
                        c1_gen[current_idx] = p2.genotype[current_idx]
                        c2_gen[current_idx] = p1.genotype[current_idx]

                    # Follow the cycle:
                    # 1. Find value in p1 at current_idx
                    value_in_p1 = p1.genotype[current_idx]
                    # 2. Find where that value is in p2
                    current_idx = p2_value_to_index[value_in_p1]
                    
            offspring.append(Individual(c1_gen))
            offspring.append(Individual(c2_gen))
            
        return offspring