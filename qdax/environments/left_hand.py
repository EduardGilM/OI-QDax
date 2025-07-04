# Copyright 2025 The QDax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Environment for the LEAP Left Hand robotic hand."""

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import os


class LeftHand(PipelineEnv):
    """
    ### Description

    This environment simulates a LEAP Left Hand robotic hand with 16 joints:
    - Index finger: 4 joints (mcp, rot, pip, dip)
    - Middle finger: 4 joints (mcp, rot, pip, dip)
    - Ring finger: 4 joints (mcp, rot, pip, dip)
    - Thumb: 4 joints (cmc, axl, mcp, ipl)

    The goal is to control the hand to perform various manipulation tasks.

    ### Action Space

    The agents take a 16-element vector for actions. The action space is a
    continuous vector with all elements in [-1.0, 1.0], where each action
    represents the normalized position command for each joint.

    | Num | Action                               | Control Min | Control Max | Joint Name |
    |-----|--------------------------------------|-------------|-------------|------------|
    | 0   | Index finger MCP joint               | -1          | 1           | if_mcp     |
    | 1   | Index finger rotation joint          | -1          | 1           | if_rot     |
    | 2   | Index finger PIP joint               | -1          | 1           | if_pip     |
    | 3   | Index finger DIP joint               | -1          | 1           | if_dip     |
    | 4   | Middle finger MCP joint              | -1          | 1           | mf_mcp     |
    | 5   | Middle finger rotation joint         | -1          | 1           | mf_rot     |
    | 6   | Middle finger PIP joint              | -1          | 1           | mf_pip     |
    | 7   | Middle finger DIP joint              | -1          | 1           | mf_dip     |
    | 8   | Ring finger MCP joint                | -1          | 1           | rf_mcp     |
    | 9   | Ring finger rotation joint           | -1          | 1           | rf_rot     |
    | 10  | Ring finger PIP joint                | -1          | 1           | rf_pip     |
    | 11  | Ring finger DIP joint                | -1          | 1           | rf_dip     |
    | 12  | Thumb CMC joint                      | -1          | 1           | th_cmc     |
    | 13  | Thumb axial joint                    | -1          | 1           | th_axl     |
    | 14  | Thumb MCP joint                      | -1          | 1           | th_mcp     |
    | 15  | Thumb IPL joint                      | -1          | 1           | th_ipl     |

    ### Observation Space

    The observation space consists of joint positions and velocities:
    - Joint positions: 16 values (one for each joint)
    - Joint velocities: 16 values (one for each joint)
    - Additional state information (palm position, orientation, etc.)

    Total observation space: 38 dimensions
    - 16 joint positions
    - 16 joint velocities
    - 3 palm position (x, y, z)
    - 3 palm linear velocity (x, y, z)

    ### Rewards

    The reward function can be customized for different tasks. Default reward includes:
    - Task-specific reward (to be defined based on the specific manipulation task)
    - Control penalty: negative reward for large actions
    - Stability reward: reward for maintaining stable pose

    ### Starting State

    All joints start at their initial positions with small random noise added
    for stochasticity.

    ### Episode Termination

    The episode terminates when the episode length is greater than 1000 steps.
    """

    def __init__(
        self,
        ctrl_cost_weight=0.01,
        reset_noise_scale=0.1,
        task_reward_weight=1.0,
        backend='generalized',
        **kwargs
    ):
        # Get the path to the left_hand.xml file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        xml_path = os.path.join(project_root, 'assets', 'left_hand.xml')
        
        # Load the MuJoCo model
        sys = mjcf.load(xml_path)

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.003125})
            n_frames = 16

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._task_reward_weight = task_reward_weight

        # Joint names for reference
        self._joint_names = [
            'if_mcp', 'if_rot', 'if_pip', 'if_dip',  # Index finger
            'mf_mcp', 'mf_rot', 'mf_pip', 'mf_dip',  # Middle finger
            'rf_mcp', 'rf_rot', 'rf_pip', 'rf_dip',  # Ring finger
            'th_cmc', 'th_axl', 'th_mcp', 'th_ipl'   # Thumb
        ]

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
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
            'palm_position_x': pipeline_state.x.pos[0, 0],
            'palm_position_y': pipeline_state.x.pos[0, 1],
            'palm_position_z': pipeline_state.x.pos[0, 2],
            'reward_ctrl': zero,
            'reward_task': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Control cost (penalty for large actions)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        # Task reward (can be customized based on specific task)
        # For now, we'll use a simple stability reward
        task_reward = self._compute_task_reward(pipeline_state)

        obs = self._get_obs(pipeline_state)
        reward = task_reward - ctrl_cost
        
        state.metrics.update(
            palm_position_x=pipeline_state.x.pos[0, 0],
            palm_position_y=pipeline_state.x.pos[0, 1],
            palm_position_z=pipeline_state.x.pos[0, 2],
            reward_task=task_reward,
            reward_ctrl=-ctrl_cost,
        )

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        # Joint positions and velocities
        joint_positions = pipeline_state.q
        joint_velocities = pipeline_state.qd
        
        # Palm position and velocity
        palm_pos = pipeline_state.x.pos[0]  # Palm is the root body
        palm_vel = pipeline_state.x.vel[0][:3]  # Linear velocity only
        
        # Concatenate all observations
        obs = jp.concatenate([
            joint_positions,
            joint_velocities,
            palm_pos,
            palm_vel
        ])
        
        return obs

    def _compute_task_reward(self, pipeline_state: base.State) -> jax.Array:
        """Compute task-specific reward.
        
        This is a placeholder implementation that can be customized for specific tasks.
        For now, it provides a simple stability reward.
        """
        # Simple stability reward: penalize large joint velocities
        joint_velocities = pipeline_state.qd
        velocity_penalty = jp.sum(jp.square(joint_velocities))
        
        # Reward for maintaining a stable pose
        stability_reward = -0.1 * velocity_penalty
        
        return self._task_reward_weight * stability_reward

    @property
    def action_size(self) -> int:
        """Return the number of actions (joints)."""
        return len(self._joint_names)

    @property
    def observation_size(self) -> int:
        """Return the size of the observation space."""
        # joint positions + joint velocities + palm position + palm velocity
        return self.sys.q_size() + self.sys.qd_size() + 3 + 3
