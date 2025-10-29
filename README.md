# evolvepy: A Modern, Strategy-Based Evolutionary Algorithm Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https_://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https_://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https_://github.com/psf/black)

`evolvepy` is a flexible, modern, and lightweight Evolutionary Algorithm (EA) framework in Python. It is designed from the ground up to be **strategy-based** and **extensible**, allowing you to build complex evolutionary behaviors by simply "plugging in" different components.

This project is built for both researchers who need to experiment with novel strategies and developers who need a robust engine to run optimizations.

## 1. What the project does

`evolvepy` provides a core `EvolutionaryAlgorithm` engine that handles the main evolutionary loop (evaluation, selection, survival). This engine is an "orchestrator" that delegates all specific logic to **injected strategy objects**.

Instead of a monolithic class, you build an algorithm by assembling these "plugins":
* **Problem:** An implementation of `BaseProblem` that defines your genotype and fitness.
* **Selection:** A `ParentSelection` strategy (e.g., `TournamentSelection`).
* **Reproduction:** A `ReproductionStrategy` (e.g., `StandardReproduction` or `MutationOnlyReproduction`) which in turn uses...
    * **Crossover:** A specific crossover operator (e.g., `OrderedCrossover`).
    * **Mutation:** A specific mutation operator (e.g., `SwapMutation`).
* **Survival:** A `SurvivorSelection` strategy (e.g., `PlusSelection` or `AgeBasedSurvivorSelection`).
* **Services:** Utilities like a `Logger` and `analysis` tools for plotting.

## 2. Why the project is useful

The key advantage of `evolvepy` is its **flexibility** through **Separation of Concerns**.

* **Test New Ideas Fast:** Want to see if `Crossover OR Mutation` works better than `Crossover + Mutation`? Just swap the `ExclusiveReproduction` strategy for the `StandardReproduction` strategy. The engine doesn't change.
* **Avoid Overfitting:** The design lets you benchmark different "recipes" of strategies across many problems to find the most robust combination, as shown in the `examples/` folder.
* **Clean & Modern:** Built with modern Python (3.10+), full type hinting, a `src/` layout, and a `pyproject.toml` build system. No legacy `setup.py`.
* **Reusable:** The framework (`src/evolvepy`) is completely decoupled from any specific problem (like the TSP demo in `examples/`). You can install and import `evolvepy` into any new project just like you would `numpy`.

## 3. How users can get started
### Recommended Installation (Stable)
Install the latest stable release by specifying the version tag (e.g., `@v0.1.0`):

You can install `evolvepy` directly from this Git repository.

### For Library Use
To use `evolvepy` as a library in your own project (e.g., a new benchmark or application):
```bash
# Installs the core library
pip install git+[https://github.com/andrea-00/evolvepy.git@v0.1.0](https://github.com/andrea-00/evolvepy.git@v0.1.0)
```

### For Development (or to run the demo)

To run the included TSP demo or to contribute to the framework, clone this repository and install it in "editable" mode:
```bash
# 1. Clone the repository
git clone [https://github.com/andrea-00/evolvepy.git](https://github.com/andrea-00/evolvepy.git)
cd evolvepy

# 2. Install in "editable" mode
# This installs the core library PLUS optional dependencies for
# plotting (analysis) and running notebooks (dev)
pip install -e ".[analysis,dev]"
```

### Quick Start: Building an EA

Here is the conceptual flow for building and running an algorithm:
```Python
from evolvepy import EvolutionaryAlgorithm, Logger, LogLevel, BaseProblem
from evolvepy.strategies import (
    TournamentSelection,
    PlusSelection,
    StandardReproduction,
    OrderedCrossover,
    SwapMutation
)

# --- 1. Define Your Problem ---
# (This class would implement BaseProblem)
# from my_problems import MyProblem
# my_problem = MyProblem(data=...)

# --- 2. Assemble Your Strategies ---
# (Using placeholders for a generic example)
my_logger = Logger(level=LogLevel.VERBOSE)
parent_sel = TournamentSelection(k=3)
survivor_sel = PlusSelection(population_size=100)
repro_strategy = StandardReproduction(
    recombination_strategy=OrderedCrossover(crossover_rate=0.9),
    mutation_strategy=SwapMutation(individual_mutation_prob=0.1)
)

# --- 3. Initialize the Engine ---
# Inject all your components into the engine's constructor
ea = EvolutionaryAlgorithm(
    fitness_function=my_problem.get_fitness_function(),
    initialization_function=my_problem.get_initializer_function(),
    parent_selection=parent_sel,
    reproduction_strategy=repro_strategy,
    survivor_selection=survivor_sel,
    population_size=100,
    logger=my_logger
)

# --- 4. Run! ---
best_individual, history = ea.run(generations=500)
```

## 4. Where Users Can Get Help

* **Full Demo:** For a complete, runnable example, please see the `examples/tsp_demo.ipynb` notebook. It shows the full workflow, including data loading and plotting.
* **Bug Reports & Feature Requests:** If you find a bug or have an idea for a new strategy, please **[open an issue](https://github.com/andrea-00/evolvepy_project/issues)** on this repository.

## 5. Who Maintains and Contributes

This project is currently maintained by **[Andrea Di Felice/andrea-00]**.

We welcome contributions! If you would like to contribute, please follow these steps:
1.  **Fork** this repository.
2.  Create a new branch (e.g., `feature/new-crossover-op`).
3.  Commit your changes.
4.  Open a **Pull Request** with a clear description of your changes.

For more details, please see our (future) `CONTRIBUTING.md` file.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for full details.

`Copyright (c) 2025 Andrea Di Felice <andrealav2901@gmail.com>`