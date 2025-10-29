# src/evolvepy/analysis/plotting.py

from typing import Dict, List, Optional

# Dependency Handling
try:
    import matplotlib.pyplot as plt # type: ignore
    import seaborn as sns           # type: ignore
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    # We don't raise the error here, but in the function call,
    # so the module can still be imported.
    pass


def plot_convergence(
    history: Dict[str, List], 
    save_path: Optional[str] = None, 
    show: bool = True
):
    """
    Generates a standard EA convergence plot from the history object.

    This plot shows:
    1. The Best Fitness per generation (line).
    2. The Mean Fitness per generation (line).
    3. The Standard Deviation as a shaded area around the mean,
       visualizing the population's fitness diversity.

    Args:
        history: The history dictionary returned by the EA engine.
                 Must contain 'generation', 'best_fitness',
                 'mean_fitness', and 'std_fitness' keys.
        save_path (str, optional): The file path to save the plot image
            (e.g., "my_plot.png"). If None, the plot is not saved.
        show (bool): Whether to display the plot interactively
            (e.g., in a Jupyter Notebook). Defaults to True.
            
    Raises:
        ImportError: If 'matplotlib' or 'seaborn' are not installed.
                     (Install with: pip install "evolve[analysis]")
    """
    
    # Dependency Check
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Plotting requires 'matplotlib' and 'seaborn'.\n"
            "Please install them by running: pip install \"evolve[analysis]\""
        )
    
    # Data Preparation
    try:
        generations = history["generation"]
        best_fitness = history["best_fitness"]
        mean_fitness = history["mean_fitness"]
        std_fitness = history["std_fitness"]
    except KeyError as e:
        raise KeyError(f"History dictionary is missing required key: {e}. "
                       "Ensure the EA engine is populating the history correctly.")
    
    # Calculate the bounds for the standard deviation band (mean ± 1 std dev)
    std_upper = [m + s for m, s in zip(mean_fitness, std_fitness)]
    std_lower = [m - s for m, s in zip(mean_fitness, std_fitness)]

    # Plot Creation
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the Best Fitness line
    ax.plot(generations, best_fitness, 
             label="Best Fitness", 
             color="C0",  # Standard blue
             linestyle='-')

    # Plot the Mean Fitness line
    ax.plot(generations, mean_fitness, 
             label="Mean Fitness", 
             color="C1",  # Standard orange
             linestyle='--')
             
    # Plot the Standard Deviation band
    ax.fill_between(generations, std_lower, std_upper, 
                     color="C1",       # Match the mean color
                     alpha=0.2,      # Use transparency
                     label="Std Deviation (±1sigma)")

    # Plot Configuration
    ax.set_title("EA Convergence Plot", fontsize=16)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    
    # Set x-axis limit to start from generation 0
    if generations:
        ax.set_xlim(left=0, right=generations[-1])
    
    # Output (Save and/or Show)
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
            
    if show:
        plt.show()
    
    # Close the figure to free up memory
    plt.close(fig)