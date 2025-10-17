"""
Script para listar todos los agentes de un repertorio y visualizar uno específico.

Muestra una tabla con todos los agentes, sus descriptores y fitness,
luego permite seleccionar uno para visualizar su trayectoria en 3D.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import argparse

from qdax.environments import create
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Params, RNGKey


def load_repertoire(repertoire_path: str):
    """Carga el repertorio desde disco."""
    print(f"\n{'='*80}")
    print(f"Cargando repertorio desde: {repertoire_path}")
    print(f"{'='*80}")
    
    centroids = np.load(os.path.join(repertoire_path, "centroids.npy"))
    descriptors = np.load(os.path.join(repertoire_path, "descriptors.npy"))
    fitnesses = np.load(os.path.join(repertoire_path, "fitnesses.npy"))
    
    genotypes_array = np.load(os.path.join(repertoire_path, "genotypes.npy"), allow_pickle=True)
    
    if genotypes_array.ndim == 0:
        genotypes = genotypes_array.item()
        genotypes_are_pytree = True
    else:
        genotypes = genotypes_array
        genotypes_are_pytree = False
    
    valid_mask = fitnesses != -np.inf
    valid_fitnesses = fitnesses[valid_mask]
    
    print(f"  Centroides: {centroids.shape}")
    print(f"  Descriptores: {descriptors.shape}")
    print(f"  Fitness: {fitnesses.shape}")
    print(f"  Genotipos: {'pytree' if genotypes_are_pytree else genotypes.shape}")
    print(f"\nEstadísticas:")
    print(f"  Soluciones válidas: {np.sum(valid_mask)} / {len(fitnesses)}")
    print(f"  Cobertura: {100 * np.mean(valid_mask):.2f}%")
    if len(valid_fitnesses) > 0:
        print(f"  Fitness máximo: {np.max(valid_fitnesses):.4f}")
        print(f"  Fitness promedio: {np.mean(valid_fitnesses):.4f}")
        print(f"  Fitness mínimo: {np.min(valid_fitnesses):.4f}")
    print(f"{'='*80}\n")
    
    return centroids, descriptors, fitnesses, genotypes, genotypes_are_pytree


def list_all_agents(descriptors: np.ndarray, fitnesses: np.ndarray, sort_by_fitness: bool = True):
    """Lista todos los agentes con sus descriptores y fitness."""
    print(f"\n{'='*80}")
    print(f"LISTA DE TODOS LOS AGENTES")
    print(f"{'='*80}\n")
    
    valid_mask = fitnesses != -np.inf
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("⚠️  No hay agentes válidos en el repertorio.")
        return []
    
    if sort_by_fitness:
        sorted_order = np.argsort(fitnesses[valid_indices])[::-1]
        valid_indices = valid_indices[sorted_order]
    
    print(f"{'Rank':<6} {'Índice':<8} {'Fitness':<12} {'Descriptor 1':<14} {'Descriptor 2':<14}")
    print(f"{'-'*6} {'-'*8} {'-'*12} {'-'*14} {'-'*14}")
    
    for rank, idx in enumerate(valid_indices, 1):
        fit = fitnesses[idx]
        desc = descriptors[idx]
        
        if rank <= 10:
            color = '\033[92m'
        elif rank <= 50:
            color = '\033[93m'
        else:
            color = '\033[0m'

        print(f"{color}{rank:<6} {idx:<8} {fit:<12.4f} {desc[0]:<14.4f} {desc[1]:<14.4f}\033[0m")
    
    print(f"\n{'='*80}")
    print(f"Total de agentes válidos: {len(valid_indices)}")
    print(f"{'='*80}\n")
    
    return valid_indices


def extract_policy_params(genotypes, index: int, genotypes_are_pytree: bool, 
                         policy_network, env, seed: int):
    """Extrae los parámetros de la política para un índice dado."""
    if genotypes_are_pytree:
        return jax.tree_map(lambda x: x[index], genotypes)
    else:
        random_key = jax.random.PRNGKey(seed + index)
        fake_obs = jnp.zeros(shape=(env.observation_size,))
        template_params = policy_network.init(random_key, fake_obs)
        
        flat_template, tree_def = jax.tree_util.tree_flatten(template_params)
        flat_genotype = genotypes[index]
        
        split_genotype = []
        offset = 0
        for param in flat_template:
            size = param.size
            split_genotype.append(jnp.array(flat_genotype[offset:offset+size]).reshape(param.shape))
            offset += size
        
        return jax.tree_util.tree_unflatten(tree_def, split_genotype)


def rollout_policy(env, policy_network, policy_params: Params, 
                  seed: int, episode_length: int) -> Tuple[List[np.ndarray], List[float]]:
    """Ejecuta una política y recolecta la trayectoria."""
    random_key = jax.random.PRNGKey(seed)
    reset_key, random_key = jax.random.split(random_key)
    env_state = env.reset(reset_key)
    
    positions = []
    rewards = []
    
    if hasattr(env_state, 'info') and 'pos' in env_state.info:
        positions.append(np.array(env_state.info['pos']))
    
    # Rollout
    for step in range(episode_length):
        action = policy_network.apply(policy_params, env_state.obs)
        env_state = env.step(env_state, action)
        
        if hasattr(env_state, 'info') and 'pos' in env_state.info:
            positions.append(np.array(env_state.info['pos']))
        rewards.append(float(env_state.reward))
    
    return positions, rewards


def create_function_surface(function_type: str, x_range, y_range, minval: float, maxval: float, 
                           n_dimensions: int, episode_length: int, z_fixed=0.0):
    """Crea la superficie de la función objetivo.
    
    Para sphere: Mostramos el valle clásico (minimización visual) pero con colores
    que reflejan que el fitness guardado es alto=mejor (maximización en rewards).
    """
    X, Y = np.meshgrid(x_range, y_range)
    
    if function_type == 'sphere':
        Z_classic_sphere = (X + minval * 0.4)**2 + (Y + minval * 0.4)**2 + z_fixed**2

        max_z = ((maxval + minval * 0.4)**2 + (maxval + minval * 0.4)**2 + z_fixed**2) * n_dimensions
        min_z = 0  

        Z = 100 - (Z_classic_sphere / max_z * 100)
        
    else:  # rastrigin
        A = 10
        n = n_dimensions
        Z_raw = -(A * n + (X**2 - A * np.cos(2 * np.pi * X)) + 
                  (Y**2 - A * np.cos(2 * np.pi * Y)) + z_fixed**2)
        
        rastrigin_scoring = lambda x: -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
        worst_fitness_step = rastrigin_scoring(np.ones(n_dimensions) * maxval)
        best_fitness_step = rastrigin_scoring(np.zeros(n_dimensions))

        Z = (Z_raw - worst_fitness_step) * 100 / (best_fitness_step - worst_fitness_step)
    
    return X, Y, Z


def visualize_3d(positions: List[np.ndarray], fitness: float, agent_idx: int,
                function_type: str, n_dimensions: int, minval: float, maxval: float):
    """Visualiza la trayectoria en 3D."""
    if n_dimensions < 2:
        print("⚠️  Se requieren al menos 2 dimensiones para visualización 3D")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions_array = np.array(positions)
    x_traj = positions_array[:, 0]
    y_traj = positions_array[:, 1]

    z_fixed = 0.0
    if n_dimensions >= 3:
        z_fixed = np.mean(positions_array[:, 2])

    x_range = np.linspace(minval, maxval, 100)
    y_range = np.linspace(minval, maxval, 100)
    X, Y, Z = create_function_surface(function_type, x_range, y_range, minval, maxval, n_dimensions, 1, z_fixed)
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='coolwarm',
        alpha=0.6,
        edgecolor='none'
    )
    
    z_traj = []
    
    if function_type == 'sphere':
        max_z = ((maxval + minval * 0.4)**2 + (maxval + minval * 0.4)**2 + z_fixed**2) * n_dimensions
        
        for x, y in zip(x_traj, y_traj):
            z_classic = (x + minval * 0.4)**2 + (y + minval * 0.4)**2 + z_fixed**2
            z_normalized = 100 - (z_classic / max_z * 100)
            z_traj.append(z_normalized)
    else:  # rastrigin
        A = 10
        n = n_dimensions
        rastrigin_scoring = lambda x: -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
        worst_fitness_step = rastrigin_scoring(np.ones(n_dimensions) * maxval)
        best_fitness_step = rastrigin_scoring(np.zeros(n_dimensions))
        
        for x, y in zip(x_traj, y_traj):
            z_raw = -(A * n + (x**2 - A * np.cos(2 * np.pi * x)) + 
                     (y**2 - A * np.cos(2 * np.pi * y)) + z_fixed**2)
            z_normalized = (z_raw - worst_fitness_step) * 100 / (best_fitness_step - worst_fitness_step)
            z_traj.append(z_normalized)
    
    z_traj = np.array(z_traj)

    ax.plot(
        x_traj, y_traj, z_traj,
        'r-',
        linewidth=2,
        alpha=0.5,
        label=f'Trayectoria (Agente {agent_idx})',
        zorder=100
    )
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(x_traj)))
    
    overlap_threshold = 0.15
    
    for i in range(len(x_traj)):
        is_overlapping = False
        for j in range(i):
            dist = np.sqrt((x_traj[i] - x_traj[j])**2 + 
                          (y_traj[i] - y_traj[j])**2 + 
                          (z_traj[i] - z_traj[j])**2)
            if dist < overlap_threshold:
                is_overlapping = True
                break
        
        size = 30 + (i / len(x_traj)) * 50
        
        marker = 'X' if is_overlapping else 'o'
        linewidth = 1.5 if is_overlapping else 0.5 
        
        ax.scatter(
            [x_traj[i]], [y_traj[i]], [z_traj[i]],
            c=[colors[i]],
            s=size,
            marker=marker,
            zorder=100 + i,
            alpha=0.7,
            edgecolors='black',
            linewidths=linewidth
        )

    ax.scatter(
        [x_traj[0]], [y_traj[0]], [z_traj[0]],
        c='lime',
        s=300,
        marker='o',
        label='Inicio',
        zorder=200,
        edgecolors='black',
        linewidths=3
    )
    
    ax.scatter(
        [x_traj[-1]], [y_traj[-1]], [z_traj[-1]],
        c='red',
        s=300,
        marker='X',
        label='Fin',
        zorder=200,
        edgecolors='black',
        linewidths=3
    )
    
    ax.set_xlabel('X (Dimensión 1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Dimensión 2)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Fitness Normalizado (0-100)', fontsize=12, fontweight='bold')
    
    title = (f'Trayectoria del Agente {agent_idx} ({len(positions)} pasos)\n'
            f'Función: {function_type.capitalize()}, '
            f'Fitness Normalizado: {fitness:.2f}/100\n'
            f'Color: Azul (inicio) → Rojo (fin) | X = Puntos superpuestos')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Listar y visualizar agentes específicos de repertorios'
    )
    parser.add_argument(
        '--repertoire-path',
        type=str,
        required=True,
        help='Ruta al directorio del repertorio'
    )
    parser.add_argument(
        '--function',
        type=str,
        choices=['sphere', 'rastrigin'],
        required=True,
        help='Tipo de función (sphere o rastrigin)'
    )
    parser.add_argument(
        '--agent-index',
        type=int,
        default=None,
        help='Índice del agente a visualizar (opcional, se preguntará si no se proporciona)'
    )
    parser.add_argument(
        '--n-dimensions',
        type=int,
        default=3,
        help='Número de dimensiones del entorno (default: 3)'
    )
    parser.add_argument(
        '--episode-length',
        type=int,
        default=30,
        help='Longitud del episodio (default: 30)'
    )
    parser.add_argument(
        '--minval',
        type=float,
        default=-5.12,
        help='Valor mínimo del espacio (default: -5.12)'
    )
    parser.add_argument(
        '--maxval',
        type=float,
        default=5.12,
        help='Valor máximo del espacio (default: 5.12)'
    )
    parser.add_argument(
        '--policy-hidden-sizes',
        type=int,
        nargs='+',
        default=[124, 124],
        help='Tamaños de capas ocultas (default: 124 124)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria (default: 42)'
    )
    parser.add_argument(
        '--sort-by-fitness',
        action='store_true',
        default=True,
        help='Ordenar agentes por fitness (default: True)'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Solo listar agentes sin visualizar'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.repertoire_path):
        print(f"Error: No se encontró el repertorio en {args.repertoire_path}")
        return
    
    centroids, descriptors, fitnesses, genotypes, genotypes_are_pytree = load_repertoire(args.repertoire_path)
    
    valid_indices = list_all_agents(descriptors, fitnesses, args.sort_by_fitness)
    
    if len(valid_indices) == 0:
        return
    
    if args.list_only:
        return
    
    if args.agent_index is not None:
        agent_idx = args.agent_index
        if fitnesses[agent_idx] == -np.inf:
            print(f"\n⚠️  Error: El agente {agent_idx} no tiene solución válida (fitness = -inf)")
            return
    else:
        while True:
            try:
                user_input = input(f"\n¿Qué agente quieres visualizar? (Índice 0-{len(fitnesses)-1}, o 'q' para salir): ")
                if user_input.lower() == 'q':
                    print("Saliendo...")
                    return
                
                agent_idx = int(user_input)
                if agent_idx < 0 or agent_idx >= len(fitnesses):
                    print(f"⚠️  Error: Índice fuera de rango. Debe estar entre 0 y {len(fitnesses)-1}")
                    continue
                
                if fitnesses[agent_idx] == -np.inf:
                    print(f"⚠️  Error: El agente {agent_idx} no tiene solución válida (fitness = -inf)")
                    continue
                
                break
            except ValueError:
                print("⚠️  Error: Por favor ingresa un número válido o 'q' para salir")
                continue
    
    print(f"\n{'='*80}")
    print(f"Configurando entorno y generando trayectoria...")
    print(f"{'='*80}\n")
    
    env_name = f"{args.function}_oi"
    env = create(
        env_name,
        n_dimensions=args.n_dimensions,
        episode_length=args.episode_length,
        minval=args.minval,
        maxval=args.maxval,
        fixed_init_state=False,
        qdax_wrappers_kwargs=[{
            "episode_length": args.episode_length
        }]
    )
    
    policy_layer_sizes = tuple(args.policy_hidden_sizes) + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    
    policy_params = extract_policy_params(
        genotypes, agent_idx, genotypes_are_pytree,
        policy_network, env, args.seed
    )
    
    positions, rewards = rollout_policy(
        env, policy_network, policy_params,
        args.seed, args.episode_length
    )
    
    print(f"\nAgente {agent_idx}:")
    print(f"  Fitness: {fitnesses[agent_idx]:.4f}")
    print(f"  Descriptor: [{descriptors[agent_idx][0]:.4f}, {descriptors[agent_idx][1]:.4f}]")
    print(f"  Pasos: {len(positions)}")
    print(f"  Reward total: {sum(rewards):.4f}")
    print(f"  Reward promedio: {np.mean(rewards):.4f}\n")
    
    # Visualizar
    visualize_3d(
        positions, fitnesses[agent_idx], agent_idx,
        args.function, args.n_dimensions, args.minval, args.maxval
    )


if __name__ == "__main__":
    main()
