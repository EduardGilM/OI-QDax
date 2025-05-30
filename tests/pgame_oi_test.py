import functools
from typing import Any, Dict, Tuple
import os

import jax
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt

from qdax import environments
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax_envs import scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting_utils import plot_2d_map_elites_repertoire, plot_oi_map_elites_results


def run_pgame_oi_test(env_name: str = "halfcheetah_oi", num_iterations: int = 100) -> MapElitesRepertoire:
    """
    Run PGA-ME with the LZ76 (OI) wrapper.
    
    Args:
        env_name: Name of the environment with OI wrapper
        num_iterations: Number of iterations to run the algorithm
        
    Returns:
        The final repertoire
    """
    episode_length = 100
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 50000
    num_centroids = 1024
    min_bd = (0, -1)
    max_bd = (1, 1)

    # PGA-ME Emitter parameters
    proportion_mutation_ga = 0.5

    # TD3 params
    env_batch_size = 100
    replay_buffer_size = 100000
    critic_hidden_layer_size = (64, 64)
    critic_learning_rate = 3e-4
    greedy_learning_rate = 3e-4
    policy_learning_rate = 1e-3
    noise_clip = 0.5
    policy_noise = 0.2
    discount = 0.99
    reward_scaling = 1.0
    transitions_batch_size = 32
    soft_tau_update = 0.005
    num_critic_training_steps = 5
    num_pg_training_steps = 5
    policy_delay = 2

    # Init environment with OI wrapper
    env = environments.create(
        env_name,
        episode_length=episode_length,
        fixed_init_state=True,
        qdax_wrappers_kwargs=[{
            "episode_length": episode_length
        }]
    )

    # Init a random key
    random_key = jax.random.PRNGKey(seed)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=env_batch_size)
    fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the function to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated state and the transition.
        """
        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    metrics_function = functools.partial(default_qd_metrics, qd_offset=reward_offset)

    # Define the PG-emitter config
    pga_emitter_config = PGAMEConfig(
        env_batch_size=env_batch_size,
        batch_size=transitions_batch_size,
        proportion_mutation_ga=proportion_mutation_ga,
        critic_hidden_layer_size=critic_hidden_layer_size,
        critic_learning_rate=critic_learning_rate,
        greedy_learning_rate=greedy_learning_rate,
        policy_learning_rate=policy_learning_rate,
        noise_clip=noise_clip,
        policy_noise=policy_noise,
        discount=discount,
        reward_scaling=reward_scaling,
        replay_buffer_size=replay_buffer_size,
        soft_tau_update=soft_tau_update,
        num_critic_training_steps=num_critic_training_steps,
        num_pg_training_steps=num_pg_training_steps,
        policy_delay=policy_delay,
    )

    # Get the emitter
    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)

    pg_emitter = PGAMEEmitter(
        config=pga_emitter_config,
        policy_network=policy_network,
        env=env,
        variation_fn=variation_fn,
    )

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=env_batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=pg_emitter,
        metrics_function=metrics_function,
    )

    # Initialize
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    print(f"Running PGA-ME with {env_name} environment for {num_iterations} iterations...")

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

    # Plotting
    env_steps = jnp.arange(num_iterations) * episode_length * env_batch_size
    
    # Visualize results
    fig1, axes = plot_oi_map_elites_results(
        env_steps=env_steps,
        metrics=metrics,
        repertoire=repertoire,
        min_bd=min_bd,  
        max_bd=max_bd,  
    )
    
    plt.show(block=False)

    fig2, ax = plt.subplots(figsize=(10, 10))
    plot_2d_map_elites_repertoire(
        repertoire=repertoire,
        ax=ax,
        min_bd=min_bd,  
        max_bd=max_bd,    
        title=f"PGA-ME Final Archive - {env_name}"
    )
    plt.show(block=False)
    
    return repertoire


@pytest.mark.parametrize(
    "env_name",
    [
        "halfcheetah_oi",
    ],
)
def test_pgame_with_oi_wrapper(env_name: str) -> None:
    """Test function for pytest."""
    # Run with a small number of iterations for testing
    repertoire = run_pgame_oi_test(env_name, num_iterations=5)
    assert repertoire is not None
    
    # Save the repertoire
    repertoire_path = "./pgame_oi_repertoire/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)


if __name__ == "__main__":
    # Run with more iterations when executed as a script
    repertoire = run_pgame_oi_test("halfcheetah_oi", num_iterations=1000)
    
    # Save the repertoire
    repertoire_path = "./pgame_oi_repertoire/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)
    
    plt.show()  # Keep plots open until user closes them
