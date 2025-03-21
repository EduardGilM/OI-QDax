import math
from typing import Dict, Optional, Tuple
import jax.lax as lax

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
    
def euclidean_distances(X, Y=None):
    """Compute pairwise Euclidean distances between points in X and Y using JAX.
    
    Args:
        X: array of shape (n_samples_X, n_features)
        Y: array of shape (n_samples_Y, n_features), optional
    
    Returns:
        distances: array of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    
    # Compute squared Euclidean distances
    XX = jnp.sum(X * X, axis=1)[:, jnp.newaxis]
    YY = jnp.sum(Y * Y, axis=1)[jnp.newaxis, :]
    distances = XX + YY - 2 * jnp.dot(X, Y.T)
    
    # Ensure no negative values due to numerical errors
    distances = jnp.maximum(distances, 0.0)
    
    return jnp.sqrt(distances)

def k_nearest_distances(X, k=1):
    """Find k-nearest neighbors distances using JAX.
    
    Args:
        X: array of shape (n_samples, n_features)
        k: number of neighbors (including self)
    
    Returns:
        knn_distances: array of shape (n_samples, k)
    """
    # Compute pairwise distances
    dist_matrix = euclidean_distances(X)
    
    # Replace diagonal (self-distance) with infinity
    n_samples = X.shape[0]
    dist_matrix = dist_matrix.at[jnp.arange(n_samples), jnp.arange(n_samples)].set(jnp.inf)
    
    # Sort distances along rows and take k smallest values
    return jnp.sort(dist_matrix, axis=1)[:, :k]

def k_l_entropy(data, k=1):
    """Calculate entropy estimate using k-nearest neighbors with pure JAX.
    
    Args:
        data: array of shape (n_samples, n_dimensions)
        k: number of neighbors
    
    Returns:
        entropy: float, entropy estimate
    """
    n_samples, n_dimensions = data.shape
    
    # Volume of unit hypersphere in n_dimensions
    vol_hypersphere = jnp.pi**(n_dimensions/2) / math.gamma((n_dimensions/2) + 1)
    
    # Get distances to k nearest neighbors (excluding self)
    distances = k_nearest_distances(data, k)
    
    # Get the k-th nearest neighbor distance for each point
    epsilon = distances[:, k-1]
    
    # Calculate entropy using the KL estimator formula
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
        # Extract the element at row_idx, col_idx
        element = lax.dynamic_slice(matrix[row_idx], (col_idx,), (1,))
        return element[0]  # Remove extra dimension
    
    # Apply to all rows
    column_data = jax.vmap(get_element)(jnp.arange(rows))
    
    # Reshape to column vector [rows, 1]
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
    
    # Create a mask for all columns except the one to exclude
    col_indices = jnp.arange(cols)
    mask = jnp.not_equal(col_indices, col_idx)
    
    # Create result matrix filled with zeros
    result = jnp.zeros((rows, cols-1), dtype=matrix.dtype)
    
    # Copy columns one by one, skipping the excluded column
    def body_fun(i, result_matrix):
        # Skip the excluded column
        src_col = i + (i >= col_idx)  # Source column index in the original matrix
        # Only copy if it's not the excluded column
        valid_col = i < (cols - 1)
        
        # Get the column data
        col_data = extract_single_column(matrix, src_col)
        
        # Update the result matrix conditionally
        return result_matrix.at[:, i].set(
            jnp.where(valid_col, col_data.flatten(), result_matrix[:, i])
        )
    
    # Apply the loop
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
            explained_variance = pca_state.explained_variance

            total_variance = jnp.sum(explained_variance)
            explained_variance_ratio = explained_variance / total_variance
            cumulative = jnp.cumsum(explained_variance_ratio)

            n_components = jnp.sum(cumulative < 0.85) + 1

            transformed_obs = pcax.transform(pca_state, obs_seq)

            rows, cols = transformed_obs.shape

            col_indices = jnp.arange(cols)
            mask = jnp.less(col_indices, n_components)

            mask_expanded = mask.reshape(1, -1)

            reduced_obs = transformed_obs * mask_expanded
            
            # Calculate complexity and O-information
            obs_binary = action_to_binary_padded(reduced_obs)
            new_complexity = jnp.float32(LZ76_jax(obs_binary))
            new_o_info = jnp.float32(self._compute_o_information(reduced_obs))
            
            return new_complexity, new_o_info, jnp.array([new_complexity, new_o_info])
        
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
        # Transpose to get (samples, features) format
        obs_t = jnp.transpose(obs_sequence)
        n_samples, n_vars = obs_t.shape
        k = 3

        # Calculate joint entropy using all variables
        h_joint = k_l_entropy(obs_t, k)
        
        # Calculate sum of conditional terms in a JAX-friendly way
        def body_fun(j, acc):
            # Extract single column j using dynamic slicing
            column_j = extract_single_column(obs_t, j)
            
            # Calculate entropy of column j
            h_xj = k_l_entropy(column_j, 1)
            
            # Create data excluding column j
            data_excl_j = exclude_column(obs_t, j)
            
            # Calculate entropy excluding column j
            h_excl_j = k_l_entropy(data_excl_j, max(k-1, 1))
            
            # Update accumulator
            return acc + (h_xj - h_excl_j)
        
        # Apply loop over all variables
        sum_term = jax.lax.fori_loop(0, n_vars, body_fun, 0.0)
        
        # Calculate final O-Information value
        o_info = (n_vars - 2) * h_joint + sum_term
        
        return jnp.float32(o_info)
