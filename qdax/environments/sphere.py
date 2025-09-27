import jax
import jax.numpy as jnp
from brax.v1.envs import Env, State
from flax import struct
from typing import Optional

@struct.dataclass
class SphereState:
    """State for the Sphere environment."""
    pipeline_state: Optional[None]  # Not used in this env
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: dict = struct.field(default_factory=dict)
    info: dict = struct.field(default_factory=dict)

class SphereEnv(Env):
    """
    A simple, purely mathematical environment where the agent's goal is to
    find the highest reward point on an N-dimensional sphere.
    """
    def __init__(
        self,
        n_dimensions: int = 20,
        episode_length: int = 100,
        minval: float = -5.12,
        maxval: float = 5.12,
        obs_radius: float = 3.0,
    ):
        super().__init__(config=None)
        self._n_dimensions = n_dimensions
        self._episode_length = episode_length
        self.minval = minval
        self.maxval = maxval
        self._obs_radius = obs_radius

    @property
    def observation_size(self) -> int:
        # 1 (center) + 2 * n_dimensions (axes)
        num_points = 1 + 2 * self._n_dimensions
        return num_points * self._n_dimensions

    @property
    def action_size(self) -> int:
        return self._n_dimensions

    def _generate_observation(self, position: jnp.ndarray) -> jnp.ndarray:
        """Generates observation points around the agent's position."""
        # Create vectors for each axis
        axis_vectors = jnp.eye(self._n_dimensions) * self._obs_radius

        # Points along positive and negative axes
        positive_points = position + axis_vectors
        negative_points = position - axis_vectors

        # Combine all points: center, positive, negative
        all_points = jnp.concatenate(
            [
                position[jnp.newaxis, :],
                positive_points,
                negative_points
            ],
            axis=0
        )
        return all_points.flatten()

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        # Generate a random initial position
        init_pos = jax.random.uniform(
            rng,
            shape=(self._n_dimensions,),
            minval=self.minval,
            maxval=self.maxval,
        )

        # Generate initial observation
        init_obs = self._generate_observation(init_pos)

        # Initial state
        state = SphereState(
            pipeline_state=None,
            obs=init_obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics={"reward": jnp.zeros(())},
            info={
                "steps": jnp.zeros((), dtype=jnp.int32),
                "pos": init_pos,
            },
        )
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        # The action is the new position, clip it to be within bounds
        new_pos = jnp.clip(action, self.minval, self.maxval)

        # Calculate reward using the sphere function
        reward = -jnp.sum((new_pos + self.minval * 0.4) * (new_pos + self.minval * 0.4))

        # Update steps and check for termination
        steps = state.info["steps"] + 1
        done = jnp.where(steps >= self._episode_length, 1.0, 0.0)

        # Generate new observation
        new_obs = self._generate_observation(new_pos)

        # Update state
        new_info = state.info | {"steps": steps, "pos": new_pos}
        state = state.replace(
            obs=new_obs,
            reward=reward,
            done=done,
            info=new_info,
        )
        state.metrics.update(reward=reward)

        return state