import math
from typing import Dict, Optional, Tuple
import jax.lax as lax
from jax.scipy.special import digamma, gamma

import flax.struct
import jax
import jax.numpy as jnp
from brax.v1 import jumpy as jp
from brax.v1.envs import Env, State, Wrapper
import pcax
from qdax.environments.lz76 import LZ76, LZ76_jax, action_to_binary_padded
import annax


class CompletedEvalMetrics(flax.struct.PyTreeNode):
    current_episode_metrics: Dict[str, jp.ndarray]
    completed_episodes_metrics: Dict[str, jp.ndarray]
    completed_episodes: jp.ndarray
    completed_episodes_steps: jp.ndarray


class CompletedEvalWrapper(Wrapper):
    """Brax env with eval metrics for completed episodes."""

    STATE_INFO_KEY = "completed_eval_metrics"

    def reset(self, rng: jp.ndarray) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=jax.tree_util.tree_map(
                jp.zeros_like, reset_state.metrics
            ),
            completed_episodes_metrics=jax.tree_util.tree_map(
                lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics
            ),
            completed_episodes=jp.zeros(()),
            completed_episodes_steps=jp.zeros(()),
        )
        reset_state.info[self.STATE_INFO_KEY] = eval_metrics
        return reset_state

    def step(self, state: State, action: jp.ndarray) -> State:
        state_metrics = state.info[self.STATE_INFO_KEY]
        if not isinstance(state_metrics, CompletedEvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info[self.STATE_INFO_KEY]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        # steps stores the highest step reached when done = True, and then
        # the next steps becomes action_repeat
        completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
            nstate.info["steps"] * nstate.done
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b, state_metrics.current_episode_metrics, nstate.metrics
        )
        completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
        completed_episodes_metrics = jax.tree_util.tree_map(
            lambda a, b: a + jp.sum(b * nstate.done),
            state_metrics.completed_episodes_metrics,
            current_episode_metrics,
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            current_episode_metrics,
            nstate.metrics,
        )

        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=current_episode_metrics,
            completed_episodes_metrics=completed_episodes_metrics,
            completed_episodes=completed_episodes,
            completed_episodes_steps=completed_episodes_steps,
        )
        nstate.info[self.STATE_INFO_KEY] = eval_metrics
        return nstate


class ClipRewardWrapper(Wrapper):
    """Wraps gym environments to clip the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(
        self,
        env: Env,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        super().__init__(env)
        self._clip_min = clip_min
        self._clip_max = clip_max

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(
            reward=jp.clip(state.reward, a_min=self._clip_min, a_max=self._clip_max)
        )

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(
            reward=jp.clip(state.reward, a_min=self._clip_min, a_max=self._clip_max)
        )


class OffsetRewardWrapper(Wrapper):
    """Wraps gym environments to offset the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(self, env: Env, offset: float = 0.0) -> None:
        super().__init__(env)
        self._offset = offset

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(reward=state.reward + self._offset)

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(reward=state.reward + self._offset)

def k_l_entropy(data, k=1):
    """Calculate entropy estimate using k-nearest neighbors with pure JAX.
    
    Args:
        data: array of shape (n_samples, n_dimensions)
        k: number of neighbors (excluding self)
    
    Returns:
        entropy: float, entropy estimate
    """
    n_samples, n_dimensions = data.shape

    vol_hypersphere = jnp.pi**(n_dimensions/2) / gamma(n_dimensions/2 + 1)

    index = annax.Index(data)
    distances, _ = index.search(data, k=k + 1)
    epsilon = distances[:, k]
    entropy = (n_dimensions * jnp.mean(jnp.log(epsilon + 1e-10)) + 
               jnp.log(vol_hypersphere) + 0.577216 + jnp.log(n_samples-1))
    
    return jnp.float32(entropy)

def extract_single_column(matrix, col_idx):
    """Extract a single column from a matrix in a JAX-safe way.
    
    Args:
        matrix: Input matrix with shape [rows, cols]
        col_idx: Column index to extract
    
    Returns:
        A column vector with shape [rows, 1]
    """
    rows, cols = matrix.shape
    
    def get_element(row_idx):
        element = lax.dynamic_slice(matrix[row_idx], (col_idx,), (1,))
        return element[0]

    column_data = jax.vmap(get_element)(jnp.arange(rows))

    return column_data.reshape(-1, 1)

def exclude_column(matrix, col_idx):
    """Create a new matrix excluding the specified column in a JAX-safe way.
    
    Args:
        matrix: Input matrix with shape [rows, cols]
        col_idx: Column index to exclude
        
    Returns:
        A matrix with shape [rows, cols-1] with col_idx removed
    """
    return jnp.delete(matrix, col_idx, axis=1)

EXPLAINED_VARIABLES = {
    "ant": 6,
    "halfcheetah": 4,
    "walker2d": 4,
    "hopper": 4,
    "humanoid": 4,
    "humanoid_w_trap": 4,
}

class LZ76Wrapper(Wrapper):
    """Wraps gym environments to add both Lempel-Ziv complexity and O-Information of the observations."""

    def __init__(self, env: Env, episode_length: int = 1000, **kwargs):
        super().__init__(env)
        self.episode_length = episode_length
        
    @property
    def behavior_descriptor_length(self):
        return 2

    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length
    
    @property
    def behavior_descriptor_limits(self):
        return (jnp.array([0.0, -1.0]), jnp.array([1.0, 1.0]))

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        
        obs_dim = state.obs.shape[0]
        state.info["obs_sequence"] = jnp.zeros((self.episode_length, obs_dim), dtype=jnp.float32)
        state.info["current_step"] = 0
        state.info["lz76_complexity"] = jnp.float32(0)
        state.info["o_info_value"] = jnp.float32(0)
        state.info["state_descriptor"] = jnp.zeros(2, dtype=jnp.float32)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        
        obs = state.obs
        current_step = state.info["current_step"]
        obs_sequence = state.info["obs_sequence"].at[current_step, :].set(obs)
        
        is_final_step = current_step == (self.episode_length - 2)
        complexities = jnp.float32(state.info["lz76_complexity"])
        o_info_values = jnp.float32(state.info["o_info_value"])
        state_descriptor = state.info["state_descriptor"]
        
        def compute_final_metrics(obs_seq):
            pca_state = pcax.fit(obs_seq, n_components=obs_seq.shape[1])
            n_components = EXPLAINED_VARIABLES[self.env.__class__.__name__.lower()]

            transformed_obs = pcax.transform(pca_state, obs_seq)
            reduced_obs = transformed_obs[:, :n_components]

            obs_binary = action_to_binary_padded(reduced_obs)
            raw_complexity = jnp.float32(LZ76_jax(obs_binary))
            raw_o_info = jnp.float32(self._compute_o_information(reduced_obs))
            
            normalized_complexity = (1/ (1 + jnp.exp(-0.22 * (raw_complexity - 310))))

            normalized_o_info = jnp.tanh(0.3571 * raw_o_info)

            #jax.debug.print("Raw complexity: {x}", x=raw_complexity)
            #jax.debug.print("Raw o-info: {x}", x=raw_o_info)
            #jax.debug.print("Normalized complexity: {x}", x=normalized_complexity)
            #jax.debug.print("Normalized o-info: {x}", x=normalized_o_info)
            
            return raw_complexity, raw_o_info, jnp.array([normalized_complexity, normalized_o_info])
        
        def keep_previous(_):
            return complexities, o_info_values, state_descriptor
        
        complexities, o_info_values, state_descriptor = jax.lax.cond(
            is_final_step,
            compute_final_metrics,
            keep_previous,
            obs_sequence
        )

        jax.debug.print("Current step: {x}", x=current_step)
        jax.debug.print("Complexity: {x}", x=complexities)
        jax.debug.print("O-Information: {x}", x=o_info_values)
        jax.debug.print("State descriptor: {x}", x=state_descriptor)

        state.info.update({
            "obs_sequence": obs_sequence,
            "current_step": current_step + 1,
            "lz76_complexity": complexities,
            "o_info_value": o_info_values,
            "state_descriptor": state_descriptor
        })

        #jax.debug.print("State info: {x}", x=state.info["state_descriptor"])

        return state
    
    def _compute_o_information(self, obs_sequence):
        """Compute O-Information with fully optimized JAX operations."""
        n_samples, n_vars = obs_sequence.shape
        k = 3
    
        # Calcular entropía conjunta
        h_joint = k_l_entropy(obs_sequence, k)
    
        # Función para calcular términos individuales
        def compute_h_terms(j, obs_sequence):
            # Extraer columna j
            column_j = extract_single_column(obs_sequence, j)
            h_xj = k_l_entropy(column_j, 1)
            
            # Excluir la columna j
            data_excl_j = exclude_column(obs_sequence, j)
            h_excl_j = k_l_entropy(data_excl_j, max(k-1, 1))
            
            return h_xj - h_excl_j
    
        # Calcular términos en paralelo
        sum_term = jnp.sum(jax.vmap(compute_h_terms, in_axes=(0, None))(jnp.arange(n_vars), obs_sequence))
    
        # Calcular O-Information
        return (n_vars - 2) * h_joint + sum_term
