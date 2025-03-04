import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Union
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire


def compute_coverage(repertoire: MapElitesRepertoire) -> float:
    """Compute the coverage of a repertoire (percentage of filled cells).

    Args:
        repertoire: A MAP-Elites repertoire

    Returns:
        The coverage percentage
    """
    filled_cells = jnp.sum(repertoire.fitnesses != -jnp.inf)
    total_cells = repertoire.fitnesses.size
    return (filled_cells / total_cells) * 100.0


def compute_metrics_from_repertoire(
    repertoire: MapElitesRepertoire
) -> Dict[str, float]:
    """Compute the main QD metrics from a repertoire.

    Args:
        repertoire: A MAP-Elites repertoire

    Returns:
        A dictionary with the computed metrics
    """

    valid_fitnesses = repertoire.fitnesses[repertoire.fitnesses != -jnp.inf]
    
    metrics = {
        "coverage": compute_coverage(repertoire),
        "max_fitness": jnp.max(valid_fitnesses) if valid_fitnesses.size > 0 else 0.0,
        "mean_fitness": jnp.mean(valid_fitnesses) if valid_fitnesses.size > 0 else 0.0,
        "qd_score": jnp.sum(valid_fitnesses) if valid_fitnesses.size > 0 else 0.0,
    }
    
    return metrics


def calculate_oi_metrics(repertoire: MapElitesRepertoire) -> Dict[str, float]:
    """
    Calculate metrics specific for OI experiments.
    
    Args:
        repertoire: A MAP-Elites repertoire with LZ76 and O-Information descriptors
        
    Returns:
        Dictionary with metrics related to OI behaviors
    """
    valid_mask = repertoire.fitnesses != -jnp.inf

    valid_descriptors = repertoire.descriptors[valid_mask]

    lz76_values = valid_descriptors[:, 0] if valid_descriptors.shape[0] > 0 else jnp.array([])
    o_info_values = valid_descriptors[:, 1] if valid_descriptors.shape[0] > 0 else jnp.array([])

    metrics = {
        "mean_lz76": jnp.mean(lz76_values) if lz76_values.size > 0 else 0.0,
        "max_lz76": jnp.max(lz76_values) if lz76_values.size > 0 else 0.0,
        "mean_o_info": jnp.mean(o_info_values) if o_info_values.size > 0 else 0.0,
        "max_o_info": jnp.max(o_info_values) if o_info_values.size > 0 else 0.0,
        "min_o_info": jnp.min(o_info_values) if o_info_values.size > 0 else 0.0,
    }
    
    return metrics


def prepare_metrics_for_plotting(
    metrics: Dict[str, Union[List, jnp.ndarray]], 
    repertoire: Optional[MapElitesRepertoire] = None,
    env_steps: jnp.ndarray = None
) -> Dict[str, jnp.ndarray]:
    """
    Prepare metrics for plotting by ensuring they have the right format.
    
    Args:
        metrics: Dictionary of metrics to process
        repertoire: Final repertoire (used to compute missing metrics)
        env_steps: Array of environment steps
        
    Returns:
        Dictionary with metrics ready for plotting
    """
    processed_metrics = dict(metrics)

    if repertoire is not None:
        final_metrics = compute_metrics_from_repertoire(repertoire)
        oi_metrics = calculate_oi_metrics(repertoire)

        final_metrics.update(oi_metrics)
        
        for key, value in final_metrics.items():
            if key not in processed_metrics:
                processed_metrics[key] = jnp.array([value])

    if env_steps is not None:
        for key in processed_metrics:
            if not isinstance(processed_metrics[key], (list, np.ndarray, jnp.ndarray)):
                processed_metrics[key] = jnp.array([processed_metrics[key]])
                
            if len(processed_metrics[key]) == 1 and len(env_steps) > 1:
                processed_metrics[key] = jnp.full_like(env_steps, processed_metrics[key][0])
    
    return processed_metrics


def plot_oi_map_elites_results(
    env_steps: jnp.ndarray,
    metrics: Dict[str, jnp.ndarray],
    repertoire: MapElitesRepertoire,
    min_bd: float = 0.0,
    max_bd: float = 1.0,
    figsize: Tuple[int, int] = (20, 10)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot the results of a MAP-Elites OI experiment with LZ76 and O-Information descriptors.
    
    Args:
        env_steps: Array of environment steps
        metrics: Dictionary of metrics to plot
        repertoire: The MAP-Elites repertoire
        min_bd: Minimum behavior descriptor value
        max_bd: Maximum behavior descriptor value
        figsize: Figure size
        
    Returns:
        Figure and axes with plots
    """
    all_metrics = prepare_metrics_for_plotting(metrics, repertoire, env_steps)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    if "coverage" in all_metrics:
        axes[0].plot(env_steps, all_metrics["coverage"])
        axes[0].set_xlabel("Environment Steps")
        axes[0].set_ylabel("Coverage (%)")
        axes[0].set_title("Coverage")
    
    if "max_fitness" in all_metrics:
        axes[1].plot(env_steps, all_metrics["max_fitness"])
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Max Fitness")  
        axes[1].set_title("Max Fitness")
    
    if "qd_score" in all_metrics:
        axes[2].plot(env_steps, all_metrics["qd_score"])
        axes[2].set_xlabel("Environment Steps")
        axes[2].set_ylabel("QD Score")
        axes[2].set_title("QD Score")

    #if "max_lz76" in all_metrics:
    #    axes[3].plot(env_steps, all_metrics["max_lz76"])
    #    axes[3].set_xlabel("Environment Steps")
    #    axes[3].set_ylabel("Max LZ76 Complexity")
    #    axes[3].set_title("LZ76 Complexity")
    #
    #if "max_o_info" in all_metrics:
    #    axes[4].plot(env_steps, all_metrics["max_o_info"])
    #    axes[4].set_xlabel("Environment Steps")
    #    axes[4].set_ylabel("Max O-Information")
    #    axes[4].set_title("O-Information")
    
    plot_2d_map_elites_repertoire(
        repertoire=repertoire,
        ax=axes[5],
        min_bd=min_bd,
        max_bd=max_bd,
        fitness_measure="fitness",
    )
    
    plt.tight_layout()
    return fig, axes


def plot_2d_map_elites_repertoire(
    repertoire: MapElitesRepertoire,
    ax: Optional[plt.Axes] = None,
    min_bd: float = 0.0,
    max_bd: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fitness_measure: str = "fitness",
    cmap: str = "viridis",
    title: str = "MAP-Elites Archive"
) -> plt.Axes:
    """
    Plot a 2D visualization of a MAP-Elites repertoire with LZ76 and O-Information as descriptors.
    
    Args:
        repertoire: The repertoire to visualize
        ax: Optional matplotlib axes
        min_bd: Minimum behavior descriptor value
        max_bd: Maximum behavior descriptor value
        vmin: Minimum fitness for colormap
        vmax: Maximum fitness for colormap
        fitness_measure: Measure to use for color ("fitness" or "density")
        cmap: Colormap name
        title: Plot title
        
    Returns:
        The matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    valid_mask = repertoire.fitnesses != -jnp.inf
    valid_fitnesses = repertoire.fitnesses[valid_mask]
    valid_descriptors = repertoire.descriptors[valid_mask]
    
    if len(valid_descriptors) == 0:
        ax.text(0.5, 0.5, "No valid solutions in repertoire", 
                horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(min_bd, max_bd)
        ax.set_ylim(min_bd, max_bd)
        ax.set_xlabel("LZ76 Complexity")
        ax.set_ylabel("O-Information")
        ax.set_title(title)
        return ax
    
    if vmin is None:
        vmin = jnp.min(valid_fitnesses)
    if vmax is None:
        vmax = jnp.max(valid_fitnesses)
    
    sc = ax.scatter(
        valid_descriptors[:, 0], 
        valid_descriptors[:, 1], 
        c=valid_fitnesses,
        cmap=cmap,
        s=20,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
    )
    

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Fitness")

    ax.set_xlim(min_bd, max_bd)
    ax.set_ylim(min_bd, max_bd)
    ax.set_xlabel("LZ76 Complexity")
    ax.set_ylabel("O-Information")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_map_elites_results(
    env_steps: jnp.ndarray,
    metrics: Dict[str, jnp.ndarray],
    repertoire: Optional[MapElitesRepertoire] = None,
    min_bd: float = 0.0,
    max_bd: float = 1.0,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Compatible wrapper for the original plot_map_elites_results function.
    
    Args:
        env_steps: Array of environment steps
        metrics: Dictionary of metrics
        repertoire: Final MAP-Elites repertoire
        min_bd: Minimum behavior descriptor value
        max_bd: Maximum behavior descriptor value
        
    Returns:
        Figure and axes with plots
    """
    return plot_oi_map_elites_results(
        env_steps=env_steps,
        metrics=metrics,
        repertoire=repertoire,
        min_bd=min_bd,
        max_bd=max_bd,
    )
