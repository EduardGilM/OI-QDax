import functools
import time
from typing import Dict, Tuple
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from qdax import environments
from qdax.core.containers.archive import score_euclidean_novelty
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.dpg_emitter import DiversityPGConfig
from qdax.core.emitters.qdpg_emitter import QDPGEmitter, QDPGEmitterConfig
from qdax.core.emitters.qpg_emitter import QualityPGConfig
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results
from qdax.utils.plotting_utils import plot_2d_map_elites_repertoire, plot_oi_map_elites_results


def run_qdpg_oi_test(env_name: str, num_iterations: int = 100) -> None:
    """
    Run QDPG with Open-Endedness metrics (OI wrapper) on a specified environment.
    This function closely follows the structure of the qdpg.ipynb notebook.
    """
    # Configuration parameters
    episode_length = 30
    seed = 42
    policy_hidden_layer_sizes = (256, 256)
    iso_sigma = 0.005
    line_sigma = 0.05
    num_init_cvt_samples = 50000
    num_centroids = 1024
    
    # Behavior descriptors bounds for OI wrapper
    min_bd = (0, -1)
    max_bd = (1, 1)

    # Batch sizes
    quality_pg_batch_size = 12
    diversity_pg_batch_size = 12
    ga_batch_size = 100
    env_batch_size = quality_pg_batch_size + diversity_pg_batch_size + ga_batch_size

    # TD3 params
    replay_buffer_size = 1000000
    critic_hidden_layer_size = (256, 256)
    critic_learning_rate = 3e-4
    greedy_learning_rate = 3e-4
    policy_learning_rate = 1e-3
    noise_clip = 0.5
    policy_noise = 0.2
    discount = 0.99
    reward_scaling = 1.0
    transitions_batch_size = 256
    soft_tau_update = 0.005
    num_critic_training_steps = 300
    num_pg_training_steps = 100
    policy_delay = 2

    archive_acceptance_threshold = 0.1
    archive_max_size = 10000

    num_nearest_neighb = 5
    novelty_scaling_ratio = 1.0

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

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=env_batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # Define the function to play a step with the policy in the environment
    def play_step_fn(
        env_state,
        policy_params,
        random_key,
    ):
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

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define the Quality PG emitter config
    qpg_emitter_config = QualityPGConfig(
        env_batch_size=quality_pg_batch_size,
        batch_size=transitions_batch_size,
        critic_hidden_layer_size=critic_hidden_layer_size,
        critic_learning_rate=critic_learning_rate,
        actor_learning_rate=greedy_learning_rate,
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

    # Define the Diversity PG emitter config
    dpg_emitter_config = DiversityPGConfig(
        env_batch_size=diversity_pg_batch_size,
        batch_size=transitions_batch_size,
        critic_hidden_layer_size=critic_hidden_layer_size,
        critic_learning_rate=critic_learning_rate,
        actor_learning_rate=greedy_learning_rate,
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
        archive_acceptance_threshold=archive_acceptance_threshold,
        archive_max_size=archive_max_size,
    )

    # Define the QDPG Emitter config
    qdpg_emitter_config = QDPGEmitterConfig(
        qpg_config=qpg_emitter_config,
        dpg_config=dpg_emitter_config,
        iso_sigma=iso_sigma,
        line_sigma=line_sigma,
        ga_batch_size=ga_batch_size,
    )

    # Get the emitter
    score_novelty = jax.jit(
        functools.partial(
            score_euclidean_novelty,
            num_nearest_neighb=num_nearest_neighb,
            scaling_ratio=novelty_scaling_ratio,
        )
    )

    # Define the QDPG emitter
    qdpg_emitter = QDPGEmitter(
        config=qdpg_emitter_config,
        policy_network=policy_network,
        env=env,
        score_novelty=score_novelty,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=qdpg_emitter,
        metrics_function=metrics_function,
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

    # Compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    # Ejecutar MAP-Elites durante el número especificado de iteraciones
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


    env_steps = jnp.arange(num_iterations) * episode_length * ga_batch_size
    
    # Visualizar resultados
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
        title=f"Archive Final - {env_name} (batch_size={ga_batch_size})"
    )
    plt.show(block=False)
    return repertoire


@pytest.mark.parametrize(
    "env_name, num_iterations",
    [
        ("halfcheetah_oi", 20),
    ],
)
def test_qdpg_oi(env_name: str, num_iterations: int) -> None:
    """Test QDPG with OI metrics."""
    repertoire = run_qdpg_oi_test(env_name, num_iterations)
    # Basic assertions to verify the test ran correctly
    assert repertoire is not None


if __name__ == "__main__":
    # For direct execution, run with more iterations on a specific environment
    repertoire= run_qdpg_oi_test("halfcheetah_oi", num_iterations=100)

    plt.show(block=True)
