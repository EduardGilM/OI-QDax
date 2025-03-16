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
    marginal_entropies = jax.vmap(lambda i: entropy_jax(jnp.array([[cov[i, i]]], dtype=jnp.float32), 1))(
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
    tc_minus_i = jax.vmap(lambda i: tc_jax(get_cov_minus_i_jax(cov, i), dim - 1))(
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
        state.info["lz76_complexity"] = 0
        state.info["o_info_value"] = 0
        state.info["state_descriptor"] = jnp.zeros(2, dtype=jnp.float32)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        
        obs = state.obs
        obs_binary_batch = action_to_binary_padded(obs)
            
        current_step = state.info["current_step"]
        obs_sequence = state.info["obs_sequence"].at[:, current_step].set(obs)
        
        if current_step == self.episode_length:
            complexities = jnp.float32(LZ76_jax(obs_binary_batch))
            o_info_values = self._compute_o_information(obs_sequence)
            state_descriptor = jnp.array([complexities, o_info_values], dtype=jnp.float32)
        else:
            complexities = 0
            o_info_values = 0
            state_descriptor = jnp.zeros(2, dtype=jnp.float32)

        state.info["obs_sequence"] = obs_sequence
        state.info["current_step"] = current_step + 1
        state.info["lz76_complexity"] = complexities
        state.info["o_info_value"] = o_info_values
        state.info["state_descriptor"] = state_descriptor

        return state
    
    def _compute_o_information(self, obs_sequence):
        """Calcula la O-Information utilizando la secuencia de observaciones"""
        cov_matrix = jnp.cov(obs_sequence)
        num_vars = cov_matrix.shape[0]
        o_info = o_inf_jax(cov_matrix, num_vars)
        return o_info
