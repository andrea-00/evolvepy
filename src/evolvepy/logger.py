# src/evolvepy/logger.py

import sys
import numpy as np
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Dict, Any, TextIO

from .individual import Individual, GenotypeType


class LogLevel(Enum):
    """
    Defines the logging verbosity levels for the engine.
    
    Attributes:
        NONE: No output will be printed.
        INFO: Only a final summary report is printed at the end.
        VERBOSE: Prints periodic updates during the run and a final summary.
    """
    NONE = auto()
    INFO = auto()
    VERBOSE = auto()


class BaseLogger(ABC):
    """
    Abstract Base Class (Interface) for all Logger objects.
    
    This defines the "contract" that the EvolutionaryAlgorithm engine
    expects from any logger it is given. It ensures all loggers
    can handle the different stages of the evolutionary process.
    """

    @abstractmethod
    def log_start(self, engine_config: Dict[str, Any]):
        """
        Called once at the beginning of the EA run.
        
        Args:
            engine_config: A dictionary containing the setup parameters
                           of the EvolutionaryAlgorithm instance.
        """
        pass

    @abstractmethod
    def log_generation(self, 
                       generation: int, 
                       history: Dict[str, list],
                       best_in_gen: Individual[GenotypeType]):
        """
        Called at the end of each generation (if the log level allows).

        Args:
            generation: The current generation number (e.g., 1, 2, ...).
            history: The history dictionary from the engine, containing
                     lists of stats ('best_fitness', 'mean_fitness', etc.).
            best_in_gen: The best individual found *in this specific* generation.
        """
        pass

    @abstractmethod
    def log_end(self,
                best_individual: Individual[GenotypeType],
                total_generations: int):
        """
        Called once at the very end of the EA run.

        Args:
            best_individual: The best individual found across all generations.
            total_generations: The total number of generations completed.
        """
        pass

    def close(self):
        """
        Called to close any open resources, like file handles.
        """
        pass


class NullLogger(BaseLogger):
    """
    A "null" logger that performs no operations.
    
    This is used as the default by the engine when no logger is
    provided (or LogLevel.NONE is set), avoiding the need for
    'if self.logger is not None' checks in the engine's main loop.
    """
    def log_start(self, engine_config: Dict[str, Any]):
        pass  # Do nothing

    def log_generation(self, 
                       generation: int, 
                       history: Dict[str, list],
                       best_in_gen: Individual[GenotypeType]):
        pass  # Do nothing

    def log_end(self,
                best_individual: Individual[GenotypeType],
                total_generations: int):
        pass  # Do nothing


class Logger(BaseLogger):
    """
    The standard, feature-complete logger for the EA framework.
    
    It handles logging to stdout (default), stderr, or a specified file
    and respects different verbosity levels.
    """

    def __init__(self, 
                 level: LogLevel = LogLevel.INFO,
                 log_file: str | None = None,
                 verbose_every_n: int = 10):
        """
        Initializes the standard logger.

        Args:
            level (LogLevel): The verbosity level (NONE, INFO, VERBOSE).
            log_file (str | None): Path to a file to write logs to.
                If None (default), logs are written to stdout.
                Can also be 'stderr'.
            verbose_every_n (int): For VERBOSE level, how often to print
                a generation update (e.g., every 10 generations).
        """
        self.level = level
        self.verbose_every_n = max(1, verbose_every_n)
        
        self._output_stream: TextIO | None = None
        self._file_handle: TextIO | None = None
        
        if self.level == LogLevel.NONE:
            # TODO: If NONE, replace this instance
            # with a NullLogger
            return

        if log_file == 'stderr':
            self._output_stream = sys.stderr
        elif log_file:
            # Open the file in write mode (creates/truncates it)
            self._file_handle = open(log_file, 'w')
            self._output_stream = self._file_handle
        else:
            # Default to standard output
            self._output_stream = sys.stdout

    def _write(self, message: str):
        """Private helper to write to the designated stream."""
        if self._output_stream:
            self._output_stream.write(message + "\n")
            # Ensure the message appears immediately (useful for logs)
            self._output_stream.flush()

    def log_start(self, engine_config: Dict[str, Any]):
        if self.level in (LogLevel.INFO, LogLevel.VERBOSE):
            self._write("--- Evolutionary Algorithm Started ---")
            self._write(f"  Level: {self.level.name}")
            # Log key configuration details
            for key, value in engine_config.items():
                # Format complex objects (like strategies) cleanly
                val_str = f"'{value}'" if isinstance(value, str) else value.__class__.__name__
                if isinstance(value, int): val_str = f"{value}"
                self._write(f"  {key}: {val_str}")
            self._write("------------------------------------------")

    def log_generation(self, 
                       generation: int, 
                       history: Dict[str, list],
                       best_in_gen: Individual[GenotypeType]):
        
        if self.level != LogLevel.VERBOSE:
            return # Only VERBOSE logs per-generation
            
        if generation % self.verbose_every_n == 0:
            # Get the latest stats from the history
            best = history["best_fitness"][-1]
            mean = history["mean_fitness"][-1]
            std = history["std_fitness"][-1]
            worst = history["worst_fitness"][-1]
            
            # Format a clean, aligned log line
            self._write(
                f"Gen {generation:<5} | "
                f"Best: {best:<10.4f} | "
                f"Mean: {mean:<10.4f} (± {std:<8.2f}) | "
                f"Worst: {worst:<10.4f}"
            )

    def log_end(self,
                best_individual: Individual[GenotypeType],
                total_generations: int):
        
        if self.level in (LogLevel.INFO, LogLevel.VERBOSE):
            self._write("--- Evolution Finished ---")
            self._write(f"  Total Generations: {total_generations}")
            self._write(f"  Best Fitness: {best_individual.fitness:.4f}")
            self._write(f"  Best Individual Age: {best_individual.age}")
            
            # Show a snippet of the best genotype for inspection
            genotype_str = str(best_individual.genotype)
            if len(genotype_str) > 75:
                genotype_str = genotype_str[:70] + "...]"
            self._write(f"  Best Genotype: {genotype_str}")
            self._write("--------------------------------")

    def close(self):
        """Closes the log file if one was opened."""
        if self._file_handle:
            self._write("--- Log File Closed ---")
            self._file_handle.close()
            self._file_handle = None
            self._output_stream = None