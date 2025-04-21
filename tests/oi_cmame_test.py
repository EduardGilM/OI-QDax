"""Tests CMA-ME implementation with OI (LZ76) wrapper"""

import functools
from typing import Dict, Tuple, Type
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from qdax import environments
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.cma_emitter import CMAEmitter
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Descriptor, ExtraScores, Fitness, RNGKey
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    scoring_function_brax_envs as scoring_function,
)
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting_utils import plot_2d_map_elites_repertoire, plot_oi_map_elites_results


def run_oi_cmame_test(env_name: str, emitter_type: Type[CMAEmitter], num_iterations: int = 100) -> None:
    """Run CMA-ME test with OI wrapper and visualization."""
    # Environment parameters
    episode_length = 30
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 50000
    num_centroids = 1024
    min_bd = (0, -1)
    max_bd = (1, 1)

    # CMA-ME parameters
    batch_size = 100
    sigma_g = 0.5
    pool_size = 3

    # Create environment with wrapper parameters
    env = environments.create(
        env_name,
        episode_length=episode_length,
        fixed_init_state=True,
        qdax_wrappers_kwargs=[{
            "episode_length": episode_length
        }]
    )

    random_key = jax.random.PRNGKey(seed)

    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    def unflatten_params(flat_params, example):
        return jax.flatten_util.ravel_pytree(example)[1](flat_params)

    # Get the shape of flattened parameters for a single agent
    example_params = jax.tree_util.tree_map(lambda x: x[0], init_variables)
    flat_example, unravel_fn = jax.flatten_util.ravel_pytree(example_params)
    genotype_dim = len(flat_example)

    # Initialize states for all agents in the batch
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # Use the predefined play_step function from QDax
    play_step_fn = make_policy_network_play_step_fn_brax(
        policy_network=policy_network,
        env=env,
    )

    # Get the behavior descriptor extractor for this environment
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]

    # Set up the scoring function
    base_scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Wrapper to convert flat parameters back to tree for evaluation
    def scoring_fn(flat_params_batch, random_key):
        # Reshape the flat batch into tree for each agent
        params_batch = jax.vmap(unflatten_params, in_axes=(0, None))(
            flat_params_batch, example_params
        )
        
        # Call the base scoring function with tree-structured parameters
        # Pass the random_key parameter that was missing before
        fitnesses, descriptors, extra_scores, random_key = base_scoring_fn(params_batch, random_key)
        
        return fitnesses, descriptors, extra_scores, random_key

    # Set up metrics function
    reward_offset = environments.reward_offset[env_name]
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=reward_offset)

    # Calculate centroids for the descriptor space
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Initialize CMA emitter
    emitter_kwargs = {
        "batch_size": batch_size,
        "genotype_dim": genotype_dim,
        "centroids": centroids,
        "sigma_g": sigma_g,
        "min_count": 1,
        "max_count": None,
    }

    emitter = emitter_type(**emitter_kwargs)

    # Wrap with pool emitter for multiple CMA states
    emitter = CMAPoolEmitter(num_states=pool_size, emitter=emitter)

    # Set up MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
    )

    # Initial population - starting with zeros
    random_key, subkey = jax.random.split(random_key)
    initial_population = jax.random.uniform(
        subkey, shape=(batch_size, genotype_dim), minval=-0.1, maxval=0.1
    )

    # Initialize the repertoire
    repertoire, emitter_state, random_key = map_elites.init(
        initial_population, centroids, random_key
    )
    
    print(f"Running CMA-ME with {emitter_type.__name__} on {env_name}...")

    # Run MAP-Elites for the specified number of iterations
    (
        repertoire,
        emitter_state,
        random_key,
    ), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    # Generate timestamp and create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = "./graficas"
    os.makedirs(plots_dir, exist_ok=True)
    
    env_steps = jnp.arange(num_iterations) * episode_length * batch_size
    
    # Save metrics plot
    fig1, axes = plot_oi_map_elites_results(
        env_steps=env_steps,
        metrics=metrics,
        repertoire=repertoire,
        min_bd=min_bd,  
        max_bd=max_bd,  
    )
    fig1.savefig(os.path.join(plots_dir, f"cma_{emitter_type.__name__}_metrics_{timestamp}.png"))
    plt.close(fig1)

    # Save archive plot
    fig2, ax = plt.subplots(figsize=(10, 10))
    plot_2d_map_elites_repertoire(
        repertoire=repertoire,
        ax=ax,
        min_bd=min_bd,  
        max_bd=max_bd,    
        title=f"Archive Final - {env_name} with {emitter_type.__name__}"
    )
    fig2.savefig(os.path.join(plots_dir, f"cma_{emitter_type.__name__}_archive_{timestamp}.png"))
    plt.close(fig2)
    
    return repertoire


@pytest.mark.parametrize(
    "env_name, emitter_type",
    [
        ("halfcheetah_oi", CMAOptimizingEmitter),
        ("halfcheetah_oi", CMARndEmitter),
        ("halfcheetah_oi", CMAImprovementEmitter),
    ],
)
def test_oi_cmame(env_name: str, emitter_type: Type[CMAEmitter]) -> None:
    """Test function for pytest."""
    repertoire = run_oi_cmame_test(env_name, emitter_type, num_iterations=10)
    assert repertoire is not None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repertoire_path = f"./repertoires/cma_oi/{timestamp}_{emitter_type.__name__}/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)


if __name__ == "__main__":
    # Run with a small number of iterations for testing
    repertoire = run_oi_cmame_test("walker2d_oi", CMAOptimizingEmitter, num_iterations=1000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repertoire_path = f"./repertoires/cma_oi/{timestamp}_CMAImprovementEmitter/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)
