import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.tasks.brax_envs import scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics


def get_mixing_emitter(batch_size: int) -> MixingEmitter:
    """Create a mixing emitter with a given batch size."""
    variation_fn = functools.partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    mixing_emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )
    return mixing_emitter


@pytest.mark.parametrize(
    "env_name, batch_size",
    [("halfcheetah_oi", 1), ("halfcheetah_oi", 10)],
)
def test_lz76_wrapper(env_name: str, batch_size: int) -> None:
    batch_size = batch_size
    env_name = env_name
    episode_length = 10
    num_iterations = 100
    seed = 42
    policy_hidden_layer_sizes = (64, 64)
    num_init_cvt_samples = 1000
    num_centroids = 50
    min_bd = 0.0
    max_bd = 1.0

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)

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
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

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

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Define emitter
    mixing_emitter = get_mixing_emitter(batch_size)

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_fn = functools.partial(default_qd_metrics, qd_offset=reward_offset)

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
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

    # Modificar para guardar métricas entre iteraciones
    all_metrics = {}
    env_steps = []
    total_steps = 0
    
    # Run the algorithm
    for i in range(num_iterations):
        (
            repertoire,
            emitter_state,
            random_key,
        ), metrics = map_elites.scan_update(
            (repertoire, emitter_state, random_key),
            (),
        )
        
        # Actualizar conteo de pasos
        total_steps += batch_size * episode_length
        env_steps.append(total_steps)
        
        # Almacenar métricas de esta iteración
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)
        
        # Imprimir progreso
        if (i + 1) % max(1, num_iterations // 10) == 0:
            print(f"Iteration {i + 1}/{num_iterations} completed")
    
    # Convertir listas de métricas a arrays para plotting
    for k in all_metrics:
        all_metrics[k] = jnp.array(all_metrics[k])
    
    # Visualizar resultados
    try:
        import matplotlib.pyplot as plt
        from qdax.utils.plotting_utils import plot_map_elites_results, calculate_oi_metrics
        
        # Convertir env_steps a array
        env_steps = jnp.array(env_steps)
        
        # Calcular métricas específicas de OI para visualización
        oi_metrics = calculate_oi_metrics(repertoire)
        print("\nLZ76 and O-Information Metrics:")
        for k, v in oi_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Crear figuras
        print("\nGenerating visualization...")
        
        # 1. Plot de métricas generales y mapa
        fig, axes = plot_map_elites_results(
            env_steps=env_steps,
            metrics=all_metrics,
            repertoire=repertoire,
            min_bd=min_bd,
            max_bd=max_bd
        )
        
        # 2. Plot del espacio de comportamiento con LZ76 vs O-Info
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        # Extraer datos del repertorio
        valid_mask = repertoire.fitnesses != -jnp.inf
        valid_fitnesses = repertoire.fitnesses[valid_mask]
        valid_descriptors = repertoire.descriptors[valid_mask]
        
        if len(valid_descriptors) > 0:
            # Scatter plot coloreado por fitness
            scatter = ax2.scatter(
                valid_descriptors[:, 0],  # LZ76
                valid_descriptors[:, 1],  # O-Info
                c=valid_fitnesses,
                cmap='viridis',
                s=50,
                alpha=0.7
            )
            
            # Añadir colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Fitness')
            
            # Configurar ejes
            ax2.set_xlabel('LZ76 Complexity')
            ax2.set_ylabel('O-Information')
            ax2.set_title(f'Behavior Space: {env_name}')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Mostrar estadísticas
            stats_text = (
                f"Max Fitness: {jnp.max(valid_fitnesses):.2f}\n"
                f"Mean Fitness: {jnp.mean(valid_fitnesses):.2f}\n"
                f"Solutions: {len(valid_fitnesses)}\n"
                f"LZ76 Range: [{jnp.min(valid_descriptors[:, 0]):.2f}, {jnp.max(valid_descriptors[:, 0]):.2f}]\n"
                f"O-Info Range: [{jnp.min(valid_descriptors[:, 1]):.2f}, {jnp.max(valid_descriptors[:, 1]):.2f}]"
            )
            
            ax2.text(
                0.02, 0.98, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        else:
            ax2.text(0.5, 0.5, "No valid solutions found", ha='center', va='center')
        
        # Guardar figuras
        figure_dir = "./figures"
        import os
        os.makedirs(figure_dir, exist_ok=True)
        
        fig_path1 = f"{figure_dir}/{env_name}_metrics.png"
        fig_path2 = f"{figure_dir}/{env_name}_behavior_space.png"
        
        fig.savefig(fig_path1, dpi=300, bbox_inches='tight')
        fig2.savefig(fig_path2, dpi=300, bbox_inches='tight')
        
        print(f"Figures saved to:\n  {fig_path1}\n  {fig_path2}")
        
        # Mostrar las figuras
        plt.show()
        
    except ImportError as e:
        print(f"Could not visualize results: {e}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    assert repertoire is not None
    return repertoire


if __name__ == "__main__":
    # Parámetros configurables para las pruebas
    ENV_NAME = "halfcheetah_oi"
    BATCH_SIZE = 10
    
    print(f"Running OI Map-Elites test with environment: {ENV_NAME}, batch size: {BATCH_SIZE}")
    
    # Ejecutar la prueba y obtener el repertorio final
    final_repertoire = test_lz76_wrapper(env_name=ENV_NAME, batch_size=BATCH_SIZE)
    
    print("\nTest completed successfully")
