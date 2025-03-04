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

class LZ76Wrapper(Wrapper):
    """Wraps gym environments to add both Lempel-Ziv complexity and O-Information of the actions taken."""

    def __init__(self, env: Env, max_sequence_length: int = 1000, **kwargs):
        super().__init__(env)
        self.max_sequence_length = max_sequence_length
        self.max_action_binary_length = env.action_size * 32

        self.lz76_window = kwargs.get('lz76_window', self.max_sequence_length) 
        self.oi_window = kwargs.get('oi_window', 20)
        
        self.max_lz76 = (self.max_action_binary_length * self.max_sequence_length) / jnp.log2(self.max_action_binary_length * self.max_sequence_length)
        
        self.min_o_info = -self.env.action_size / 2.0
        self.max_o_info = self.env.action_size / 2.0
        
    @property
    def action_size(self):
        return self.env.action_size
        
    @property
    def behavior_descriptor_length(self):
        return 2 

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)

        action_sequence = jnp.zeros(
            (self.max_sequence_length, self.max_action_binary_length), dtype=jnp.uint8
        )
        action_sequence_raw = jnp.zeros(
            (self.max_sequence_length, self.action_size), dtype=jnp.float32
        )
        
        state.info["action_sequence"] = action_sequence
        state.info["action_sequence_raw"] = action_sequence_raw
        state.info["lz76_complexity"] = jnp.array(0.0, dtype=jnp.float32)
        state.info["o_info_value"] = jnp.array(0.0, dtype=jnp.float32)
        
        state.info["state_descriptor"] = jnp.array([0.01, 0.01], dtype=jnp.float32)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)

        current_step = jnp.int32(state.info["steps"] - 1)
        action_binary, _ = action_to_binary_padded(
            action, self.max_action_binary_length
        )
        
        action_sequence = state.info["action_sequence"]
        action_sequence = action_sequence.at[current_step % self.max_sequence_length].set(action_binary)
        
        action_sequence_raw = state.info["action_sequence_raw"]
        action_sequence_raw = action_sequence_raw.at[current_step % self.max_sequence_length].set(action)
        
        valid_sequence_length = jnp.minimum(current_step + 1, self.max_sequence_length)
        sequence_mask = jnp.arange(self.max_sequence_length) < valid_sequence_length
        masked_sequence = action_sequence * sequence_mask[:, jnp.newaxis]
        flattened_sequence = masked_sequence.reshape(-1)

        raw_complexity = LZ76_jax(flattened_sequence)
        normalized_complexity = jnp.clip(raw_complexity / self.max_lz76, 0.0, 1.0)

        valid_steps = jnp.minimum(current_step + 1, self.max_sequence_length)
        recent_steps = jnp.minimum(valid_steps, self.oi_window)
        
        recency_mask = jnp.arange(self.max_sequence_length) >= (valid_steps - recent_steps)
        recency_mask = recency_mask & (jnp.arange(self.max_sequence_length) < valid_steps)
        
        normalized_actions = (action_sequence_raw + 1.0) / 2.0

        o_info_raw = jnp.where(
            recent_steps > 1,
            self._compute_o_information(normalized_actions, recency_mask),
            jnp.array(0.0, dtype=jnp.float32)
        )

        o_info_norm = (o_info_raw - self.min_o_info) / (self.max_o_info - self.min_o_info)
        o_info_norm = jnp.clip(o_info_norm, 0.0, 1.0)

        state.info["action_sequence"] = action_sequence
        state.info["action_sequence_raw"] = action_sequence_raw
        state.info["lz76_complexity"] = normalized_complexity
        state.info["o_info_value"] = o_info_norm
        
        state.info["state_descriptor"] = jnp.array([
            jnp.maximum(normalized_complexity, 0.01),
            jnp.maximum(o_info_norm, 0.01)
        ])
        
        return state
    
    def _compute_o_information(self, action_sequence, valid_mask):
        """Calcula la O-Information utilizando solo acciones recientes"""
        mask = valid_mask[:, jnp.newaxis]
        masked_actions = action_sequence * mask

        cov_matrix = jnp.cov(masked_actions.T)
        cov_matrix = cov_matrix + jnp.eye(masked_actions.shape[1]) * 1e-6

        joint_entropy = 0.5 * jnp.log(jnp.linalg.det(cov_matrix))

        individual_entropies = jnp.array([
            0.5 * jnp.log(jnp.var(masked_actions[:, i]) + 1e-6)
            for i in range(masked_actions.shape[1])
        ])
        sum_individual_entropies = jnp.sum(individual_entropies)

        o_info = joint_entropy - sum_individual_entropies

        o_info = jnp.where(jnp.isnan(o_info), 0.0, o_info)
        o_info = jnp.where(jnp.isinf(o_info), 0.0, o_info)
        
        return o_info
