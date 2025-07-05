"""Brax pipeline environment for the articulated left hand.

The MJCF model originates from `assets/left_hand.xml` and is wrapped in a short
Brax-compatible XML (`assets/left_hand_brax.xml`).  The environment exposes all
16 position actuators that are defined in the MJCF (index, middle, ring &
thumb).  A very simple reward is used for now – it penalises large actions and
keeps the episode alive indefinitely.  This makes the env immediately usable in
QDax (for e.g. pure quality-diversity or goal-conditioned tasks) while letting
researchers plug in custom reward functions later on.
"""
from __future__ import annotations

from typing import Tuple

from etils import epath
from jax import numpy as jp
import jax

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

# -----------------------------------------------------------------------------
# Helper to locate the asset regardless of the installation layout.
# -----------------------------------------------------------------------------

_ASSET_PATH = epath.Path(__file__).parent.parent.parent / "assets" / "left_hand.xml"


class LeftHandBrax(PipelineEnv):
    """A very lightweight Brax environment controlling a left robotic hand."""

    def __init__(
        self,
        ctrl_cost_weight: float = 1e-3,
        reset_noise_scale: float = 0.01,
        exclude_current_positions_from_observation: bool = False,
        backend: str = "generalized",
        **kwargs,
    ) -> None:
        # ------------------------------------------------------------------
        # Load MJCF and tweak actuator gears if required.
        # ------------------------------------------------------------------
        sys = mjcf.load(_ASSET_PATH)

        # By default we keep the original MJCF parameters.  If the user wants a
        # stiffer hand they can pass a gear vector through **kwargs.
        if backend in ["spring", "positional"]:
            # The hand tends to be numerically stiff, so we reduce the timestep
            # when using the spring or positional back-end.
            sys = sys.tree_replace({"opt.timestep": 0.003125})

        # Default to 1 physics step per env.step() call if not provided.
        kwargs["n_frames"] = kwargs.get("n_frames", 1)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    # --------------------------------------------------------------------------
    # Pipelines
    # --------------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> State:  # type: ignore[override]
        """Resets the hand to a random pose close to the default."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_ctrl": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:  # type: ignore[override]
        """Advances the simulation by one step."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Simple L2 control cost (can be replaced by a task-specific reward).
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        reward = -ctrl_cost

        obs = self._get_obs(pipeline_state)
        state.metrics.update(reward_ctrl=-ctrl_cost)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    # --------------------------------------------------------------------------
    # Observation helper
    # --------------------------------------------------------------------------
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        position = pipeline_state.q
        velocity = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        return jp.concatenate((position, velocity))
