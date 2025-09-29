import os
from typing import Dict, Tuple, Type
import jax
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from qdax.core.emitters.cma_emitter import CMAEmitter
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter
from qdax.core.map_elites import MAPElites
from qdax.custom_types import Descriptor, ExtraScores, Fitness, RNGKey
from qdax.environments import create

@pytest.mark.parametrize(
    "emitter_type",
    [CMAImprovementEmitter],
)
def test_cma_me_sphere(emitter_type: Type[CMAEmitter]) -> None:
    """
    Test CMA-ME algorithm on the SphereEnv.
    This test also saves a plot of the metrics and a heatmap of the final repertoire.
    """
    num_iterations = 200
    num_dimensions = 10
    episode_length = 50
    grid_shape = (20, 20)
    batch_size = 36
    sigma_g = 0.5
    minval = -5.12
    maxval = 5.12
    pool_size = 3
    noise_level = 0.1

    # Create SphereEnv with LZ76Wrapper
    env = create(
        "sphere_oi",
        n_dimensions=num_dimensions,
        episode_length=episode_length,
        minval=minval,
        maxval=maxval,
        qdax_wrappers_kwargs=[{"episode_length": episode_length}],
    )

    # Define scoring function that runs a full episode with a noisy policy
    def scoring_function(
        x: jnp.ndarray, random_key: RNGKey
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:

        def single_scoring(genotype: jnp.ndarray, sub_key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores]:
            state = env.reset(sub_key)

            def policy(obs, key):
                action = genotype + jax.random.normal(key, shape=genotype.shape) * noise_level
                return action

            def step_fn(carry, _):
                state, total_reward, key = carry
                step_key, policy_key = jax.random.split(key)
                action = policy(state.obs, policy_key)
                next_state = env.step(state, action)
                total_reward += next_state.reward
                return (next_state, total_reward, step_key), ()

            # Split key for episode steps
            episode_key = jax.random.split(sub_key, 1)[0]
            initial_carry = (state, jnp.float32(0.0), episode_key)
            (final_state, total_reward, _), _ = jax.lax.scan(
                step_fn, initial_carry, (), length=episode_length
            )

            fitness = total_reward
            descriptor = final_state.info["state_descriptor"]

            return fitness, descriptor, {}

        keys = jax.random.split(random_key, x.shape[0])
        fitnesses, descriptors, extra_scores = jax.vmap(single_scoring)(x, keys)

        return fitnesses, descriptors, extra_scores, random_key

    # Get behavior descriptor limits from the environment
    min_bd, max_bd = env.behavior_descriptor_limits

    # Define metrics function
    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict[str, jnp.ndarray]:
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)
        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    # Initial population
    random_key = jax.random.PRNGKey(0)
    initial_population = jax.random.uniform(random_key, shape=(batch_size, num_dimensions), minval=minval, maxval=maxval)

    # Create centroids
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_bd,
        maxval=max_bd,
    )

    # Create emitter
    emitter_kwargs = {
        "batch_size": batch_size,
        "genotype_dim": num_dimensions,
        "centroids": centroids,
        "sigma_g": sigma_g,
    }
    emitter = emitter_type(**emitter_kwargs)
    emitter = CMAPoolEmitter(num_states=pool_size, emitter=emitter)

    # Create MAP-Elites algorithm
    map_elites = MAPElites(
        scoring_function=scoring_function, emitter=emitter, metrics_function=metrics_fn
    )

    # Initialize
    repertoire, emitter_state, random_key = map_elites.init(
        initial_population, centroids, random_key
    )

    # Run the algorithm
    (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    # --- Plotting ---
    os.makedirs("graficas", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot QD Score
    axes[0, 0].plot(metrics["qd_score"])
    axes[0, 0].set_title("QD Score")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Score")

    # Plot Max Fitness
    axes[0, 1].plot(metrics["max_fitness"])
    axes[0, 1].set_title("Max Fitness")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Fitness")

    # Plot Coverage
    axes[1, 0].plot(metrics["coverage"])
    axes[1, 0].set_title("Coverage (%)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Coverage")

    # Plot Heatmap
    fitnesses = repertoire.fitnesses.reshape(grid_shape)
    fitnesses = jnp.where(fitnesses == -jnp.inf, jnp.nan, fitnesses)
    im = axes[1, 1].pcolormesh(min_bd[0] + jnp.arange(grid_shape[0]) * (max_bd[0] - min_bd[0]) / grid_shape[0],
                              min_bd[1] + jnp.arange(grid_shape[1]) * (max_bd[1] - min_bd[1]) / grid_shape[1],
                              fitnesses.T, shading='auto')
    axes[1, 1].set_title("Final Repertoire Heatmap")
    axes[1, 1].set_xlabel("LZ Descriptor")
    axes[1, 1].set_ylabel("OI Descriptor")
    fig.colorbar(im, ax=axes[1, 1], label="Fitness")

    plt.tight_layout()
    plt.savefig(f"graficas/sphere_cmame_test_results_{emitter_type.__name__}.png")

    # Assertions for testing
    assert metrics["coverage"][-1] > 5
    assert metrics["max_fitness"][-1] > -500
    assert metrics["qd_score"][-1] > -100000

if __name__ == "__main__":
    test_cma_me_sphere(emitter_type=CMAImprovementEmitter)