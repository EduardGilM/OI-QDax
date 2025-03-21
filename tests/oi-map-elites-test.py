import functools
from typing import Tuple
import os

import jax
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation, polynomial_mutation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    scoring_function_brax_envs as scoring_function,
)
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting_utils import plot_2d_map_elites_repertoire, plot_oi_map_elites_results
import time


def get_mixing_emitter(batch_size: int) -> MixingEmitter:
    """Create a mixing emitter with a given batch size."""
    variation_fn = functools.partial(isoline_variation, iso_sigma=0.001, line_sigma=1.0)
    mutation_fn = functools.partial(polynomial_mutation, eta=128.0, minval=-1200.0, maxval=1200.0)
    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )
    return mixing_emitter


def run_map_elites_test(env_name: str, batch_size: int, num_iterations: int = 100) -> None:
    """Run MAP-Elites test with visualization."""
    episode_length = 20
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 50000
    num_centroids = 5000
    min_bd = -200
    max_bd = 200

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

    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # Usar la función de play_step predefinida de QDax
    play_step_fn = make_policy_network_play_step_fn_brax(
        policy_network=policy_network,
        env=env,
    )

    # Obtener el extractor de descriptores de comportamiento para este entorno
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]

    # Configurar la función de puntuación con el extractor de descriptores
    scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    mixing_emitter = get_mixing_emitter(batch_size)

    reward_offset = environments.reward_offset[env_name]
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=reward_offset)

    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
    )

    # Calcular centroides para el espacio de descriptores
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Inicializar el repertorio
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )
    
    print("Running MAP-Elites...")

    # Add timing to measure MAP-Elites execution
    start_time = time.time()
    
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


    env_steps = jnp.arange(num_iterations) * episode_length * batch_size
    
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
        title=f"Archive Final - {env_name} (batch_size={batch_size})"
    )
    plt.show(block=False)
    return repertoire


@pytest.mark.parametrize(
    "env_name, batch_size",
    [
        ("halfcheetah_oi", 10),
    ],
)
def test_lz76_wrapper(env_name: str, batch_size: int) -> None:
    """Test function for pytest."""
    repertoire = run_map_elites_test(env_name, batch_size, num_iterations=10)
    assert repertoire is not None


if __name__ == "__main__":
    # Ejecutar con un tamaño de lote pequeño y pocas iteraciones para pruebas
    run_map_elites_test("halfcheetah_oi", batch_size=50, num_iterations=400)
    plt.show() 
