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

def k_nearest_distances(X, k=1):
    """Find k-nearest neighbors distances using JAX.
    
    Args:
        X: array of shape (n_samples, n_features)
        k: number of neighbors (excluding self)
    
    Returns:
        knn_distances: array of shape (n_samples, k)
    """
    return jnp.argsort(jnp.sum((X[:, None, :] - X[None, :, :])**2, axis=-1), axis=-1)[:, :k]

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

    distances = k_nearest_distances(data, k)
    epsilon = distances[:, k-1]

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
    rows, cols = matrix.shape
    
    col_indices = jnp.arange(cols)
    mask = jnp.not_equal(col_indices, col_idx)
    
    result = jnp.zeros((rows, cols-1), dtype=matrix.dtype)

    def body_fun(i, result_matrix):

        src_col = i + (i >= col_idx)
        valid_col = i < (cols - 1)

        col_data = extract_single_column(matrix, src_col)
        
        return result_matrix.at[:, i].set(
            jnp.where(valid_col, col_data.flatten(), result_matrix[:, i])
        )

    result = jax.lax.fori_loop(0, cols-1, body_fun, result)
    
    return result

class LZ76Wrapper(Wrapper):
    """Wraps gym environments to add both Lempel-Ziv complexity and O-Information of the observations."""

    def __init__(self, env: Env, episode_length: int = 1000, **kwargs):
        super().__init__(env)
        self.episode_length = episode_length
        
    @property
    def behavior_descriptor_length(self):
        return 2 

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        
        obs_dim = state.obs.shape[0]
        state.info["obs_sequence"] = jnp.zeros((obs_dim, self.episode_length), dtype=jnp.float32)
        state.info["current_step"] = 0
        state.info["lz76_complexity"] = jnp.float32(0)
        state.info["o_info_value"] = jnp.float32(0)
        state.info["state_descriptor"] = jnp.zeros(2, dtype=jnp.float32)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        
        obs = state.obs
        current_step = state.info["current_step"]
        obs_sequence = state.info["obs_sequence"].at[:, current_step].set(obs)
        
        is_final_step = current_step == (self.episode_length - 2)
        complexities = jnp.float32(state.info["lz76_complexity"])
        o_info_values = jnp.float32(state.info["o_info_value"])
        state_descriptor = state.info["state_descriptor"]
        
        def compute_final_metrics(obs_seq):
            pca_state = pcax.fit(obs_seq, n_components=obs_seq.shape[0])
            n_components = 4

            transformed_obs = pcax.transform(pca_state, obs_seq)
            reduced_obs = transformed_obs[:, :n_components]
            
            obs_binary = action_to_binary_padded(reduced_obs)
            raw_complexity = jnp.float32(LZ76_jax(obs_binary))
            raw_o_info = jnp.float32(self._compute_o_information(reduced_obs))
            
            normalized_complexity = (1/ (1 + jnp.exp(-0.22 * (raw_complexity - 200))))

            normalized_o_info = jnp.tanh(0.0049 * raw_o_info)
            
            return normalized_complexity, normalized_o_info, jnp.array([normalized_complexity, normalized_o_info])
        
        def keep_previous(_):
            return complexities, o_info_values, state_descriptor
        
        complexities, o_info_values, state_descriptor = jax.lax.cond(
            is_final_step,
            compute_final_metrics,
            keep_previous,
            obs_sequence
        )

        #obs_binary = action_to_binary_padded(obs_sequence)
        #complexities = jnp.float32(LZ76_jax(obs_binary))
        #o_info_values = jnp.float32(self._compute_o_information(obs_sequence))
        #state_descriptor = jnp.array([complexities, o_info_values])

        jax.debug.print("Current step: {x}", x=current_step)
        #jax.debug.print("Complexity: {x}", x=complexities)
        #jax.debug.print("O-Information: {x}", x=o_info_values)
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
        """Compute O-Information with JAX operations."""
        obs_t = jnp.transpose(obs_sequence)
        n_samples, n_vars = obs_t.shape
        k = 3
        h_joint = k_l_entropy(obs_t, k)
        
        def body_fun(j, acc):
            column_j = extract_single_column(obs_t, j)

            h_xj = k_l_entropy(column_j, 1)

            data_excl_j = exclude_column(obs_t, j)

            h_excl_j = k_l_entropy(data_excl_j, max(k-1, 1))
            
            return acc + (h_xj - h_excl_j)
        
        sum_term = jax.lax.fori_loop(0, n_vars, body_fun, 0.0)

        o_info = (n_vars - 2) * h_joint + sum_term
        
        return jnp.float32(o_info)
