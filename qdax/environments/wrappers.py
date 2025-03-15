from typing import Dict, Optional

import flax.struct
import jax
import jax.numpy as jnp
from brax.v1 import jumpy as jp
from brax.v1.envs import Env, State, Wrapper
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

def entropy_jax(cov, dim):
    """
    Calcula la entropía de una distribución normal multivariada con JAX.
    """
    det = jnp.linalg.det(cov)
    det = jnp.where(det < 1e-10, 1e-10, det)
    return 0.5 * dim * (1.0 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.log(det)

def get_cov_minus_i_jax(cov, i):
    """
    Obtiene la matriz de covarianza excluyendo la variable en el índice i usando JAX.
    Usa una implementación compatible con JIT y trazado.
    """
    n = cov.shape[0]
    def true_fn(j):
        return j
    def false_fn(j):
        return j + 1
    indices = jax.vmap(lambda j: jax.lax.cond(j < i, true_fn, false_fn, j))(jnp.arange(n - 1))
    gathered = jax.vmap(lambda idx1: jax.vmap(lambda idx2: cov[idx1, idx2])(indices))(indices)
    return gathered

def tc_jax(cov, dim):
    """
    Calcula la correlación total (TC) usando JAX.
    """
    nb_var = cov.shape[0]
    marginal_entropies = jax.vmap(lambda i: entropy_jax(jnp.array([[cov[i, i]]], dtype=jnp.float32), dim))(
        jnp.arange(nb_var)
    )
    sum_marginal_entropies = jnp.sum(marginal_entropies)
    joint_entropy = entropy_jax(cov, dim)
    return sum_marginal_entropies - joint_entropy

def dtc_jax(cov, dim):
    """
    Calcula la correlación total dual (DTC) usando JAX.
    """
    nb_var = cov.shape[0]
    tc_all = tc_jax(cov, dim)
    tc_minus_i = jax.vmap(lambda i: tc_jax(get_cov_minus_i_jax(cov, i), dim))(
        jnp.arange(nb_var)
    )
    tc_minus_i_sum = jnp.sum(tc_minus_i)
    return (nb_var - 1) * tc_all - tc_minus_i_sum

def o_inf_jax(cov, dim):
    """
    Calcula la O-información usando JAX.
    """
    return tc_jax(cov, dim) - dtc_jax(cov, dim)

class LZ76Wrapper(Wrapper):
    """Wraps gym environments to add both Lempel-Ziv complexity and O-Information of the observations."""

    def __init__(self, env: Env, max_sequence_length: int = 1000, **kwargs):
        super().__init__(env)
        self.max_sequence_length = max_sequence_length
        self.max_obs_binary_length = env.observation_size * 32

        self.lz76_window = kwargs.get('lz76_window', self.max_sequence_length) 
        self.oi_window = kwargs.get('oi_window', 20)
        
        self.max_lz76 = (self.max_obs_binary_length * self.max_sequence_length) / jnp.log2(self.max_obs_binary_length * self.max_sequence_length)
        
        self.min_o_info = -self.env.observation_size / 2.0
        self.max_o_info = self.env.observation_size / 2.0
        
    @property
    def behavior_descriptor_length(self):
        return 2 

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        batch_size = state.obs.shape[0] if len(state.obs.shape) > 1 else 1

        obs_sequence = jnp.zeros(
            (batch_size, self.max_sequence_length, self.max_obs_binary_length), dtype=jnp.uint8
        )
        obs_sequence_raw = jnp.zeros(
            (batch_size, self.max_sequence_length, self.env.observation_size), dtype=jnp.float32
        )
        
        state.info["obs_sequence"] = obs_sequence
        state.info["obs_sequence_raw"] = obs_sequence_raw
        state.info["lz76_complexity"] = jnp.zeros(batch_size, dtype=jnp.float32)
        state.info["o_info_value"] = jnp.zeros(batch_size, dtype=jnp.float32)
        
        state.info["state_descriptor"] = jnp.ones((batch_size, 2), dtype=jnp.float32) * 0.01
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        
        obs = state.obs
        if len(obs.shape) == 1:
            obs = obs[None, :]
        
        batch_size = obs.shape[0]
        current_step = jnp.int32(state.info["steps"] - 1)
 
        obs_binary_batch = jax.vmap(lambda o: action_to_binary_padded(o, self.max_obs_binary_length)[0])(obs)
        
        obs_sequence = state.info["obs_sequence"]
        obs_sequence_raw = state.info["obs_sequence_raw"]
        
        def update_sequence(seq, new_val, step):
            return seq.at[step % self.max_sequence_length].set(new_val)
        
        obs_sequence = jax.vmap(update_sequence)(
            obs_sequence,
            obs_binary_batch,
            jnp.full(obs_binary_batch.shape[0], current_step)
        )
        
        obs_sequence_raw = jax.vmap(update_sequence)(
            obs_sequence_raw,
            obs,
            jnp.full(obs.shape[0], current_step)
        )

        valid_sequence_length = jnp.minimum(current_step + 1, self.max_sequence_length)
        sequence_mask = jnp.arange(self.max_sequence_length) < valid_sequence_length
        
        complexities = jax.vmap(lambda seq: LZ76_jax(seq.reshape(-1)))(
            obs_sequence * sequence_mask[None, :, None]
        )
        normalized_complexities = jnp.clip(complexities / self.max_lz76, 0.0, 1.0)

        valid_steps = jnp.minimum(current_step + 1, self.max_sequence_length)
        recent_steps = jnp.minimum(valid_steps, self.oi_window)
        
        recency_mask = jnp.arange(self.max_sequence_length) >= (valid_steps - recent_steps)
        recency_mask = recency_mask & (jnp.arange(self.max_sequence_length) < valid_steps)
        
        normalized_obs = (obs_sequence_raw + 1.0) / 2.0

        o_info_values = jax.vmap(lambda obs, mask: jnp.where(
            recent_steps > 1,
            self._compute_o_information(obs, mask),
            jnp.array(0.0, dtype=jnp.float32)
        ))(normalized_obs, jnp.broadcast_to(recency_mask, normalized_obs.shape[:2]))

        o_info_norm = (o_info_values - self.min_o_info) / (self.max_o_info - self.min_o_info)
        o_info_norm = jnp.clip(o_info_norm, 0.0, 1.0)

        state_descriptor = jnp.stack([
            jnp.maximum(normalized_complexities, 0.01),
            jnp.maximum(o_info_norm, 0.01)
        ], axis=1)

        state.info["obs_sequence"] = obs_sequence
        state.info["obs_sequence_raw"] = obs_sequence_raw
        state.info["lz76_complexity"] = normalized_complexities
        state.info["o_info_value"] = o_info_norm
        state.info["state_descriptor"] = state_descriptor.reshape(batch_size, 2)
        
        return state
    
    def _compute_o_information(self, obs_sequence, valid_mask):
        """Calcula la O-Information utilizando solo observaciones recientes"""
        mask = valid_mask[:, jnp.newaxis]
        masked_obs = obs_sequence * mask

        cov_matrix = jnp.cov(masked_obs.T)
        cov_matrix = cov_matrix + jnp.eye(masked_obs.shape[1]) * 1e-6

        num_vars = cov_matrix.shape[0]
        o_info = o_inf_jax(cov_matrix, num_vars)

        o_info = jnp.where(jnp.isnan(o_info), 0.0, o_info)
        o_info = jnp.where(jnp.isinf(o_info), 0.0, o_info)
        
        return o_info
