"""Left Hand environment for manipulation tasks.
Based on the LEAP Hand robot model for quality-diversity experiments.
"""

from typing import Any, Dict
import os

import brax.v1 as brax
from brax.v1 import jumpy as jp
from brax.v1.envs import Env, State
from brax.v1.physics import bodies
import jax.numpy as jnp
import jax

# Fix for JAX compatibility issues
try:
    # Try to access the problematic attribute
    jax.core.thread_local_state
except AttributeError:
    # If it doesn't exist, create a mock version
    class MockThreadLocalState:
        def __init__(self):
            self.trace_state = MockTraceState()
    
    class MockTraceState:
        def __init__(self):
            self.trace_stack = []
    
    # Monkey patch the missing attribute
    jax.core.thread_local_state = MockThreadLocalState()


class LeftHand(Env):
    """Environment for the LEAP Left Hand robot.
    
    The robot has 16 joints:
    - Index finger: 4 joints (mcp, rot, pip, dip)
    - Middle finger: 4 joints (mcp, rot, pip, dip)
    - Ring finger: 4 joints (mcp, rot, pip, dip)
    - Thumb: 4 joints (cmc, axl, mcp, ipl)
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        # Load the left hand XML configuration
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "left_hand.xml")
        
        # Read the XML file content
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Extract just the MuJoCo config without the XML wrapper
        # Find the content between <mujoco> tags
        start_tag = config_content.find('<mujoco')
        end_tag = config_content.find('</mujoco>') + len('</mujoco>')
        
        if start_tag != -1 and end_tag != -1:
            mujoco_content = config_content[start_tag:end_tag]
        else:
            raise ValueError("Could not find valid MuJoCo configuration in left_hand.xml")
        
        # Convert XML to Brax config format
        # For now, we'll use a simplified physics config based on the humanoid
        config = _SIMPLE_HAND_CONFIG
        
        super().__init__(config=config, **kwargs)
        
        # Initialize physics properties
        body = bodies.Body(self.sys.config)
        # Skip the floor body if present
        if len(body.idx) > 16:  # hand bodies + floor
            body = jp.take(body, body.idx[:-1])  # skip the floor body
        
        self.mass = body.mass.reshape(-1, 1)
        self.inertia = body.inertia

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        
        # Initialize joint positions with small random variations
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -0.1, 0.1
        )
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -0.1, 0.1)
        
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info, jp.zeros(self.action_size))
        
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_action": zero,
            "reward_stability": zero,
            "reward_manipulation": zero,
        }
        
        return State(qp, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info, action)

        # Reward calculation for hand manipulation
        # Encourage smooth movements and penalize high torques
        action_cost = 0.01 * jp.sum(jp.square(action))
        
        # Reward for maintaining stability (palm position)
        palm_pos = qp.pos[0]  # palm is typically the first body
        stability_reward = -0.1 * jp.sum(jp.square(palm_pos[:2]))  # penalize x,y movement
        
        # Basic manipulation reward - encourage finger movement coordination
        finger_positions = qp.pos[1:6]  # first 5 finger segments
        manipulation_reward = 0.1 * jp.sum(jp.abs(action))  # reward for movement
        
        reward = manipulation_reward + stability_reward - action_cost

        # Simple termination condition - if palm falls too low
        done = jp.where(palm_pos[2] < 0.05, jp.float32(1), jp.float32(0))
        
        state.metrics.update(
            reward_action=-action_cost,
            reward_stability=stability_reward,
            reward_manipulation=manipulation_reward,
        )

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    def _get_obs(self, qp: brax.QP, info: brax.Info, action: jp.ndarray) -> jp.ndarray:
        """Get observations from the hand state."""
        # Joint observations
        joint_obs = [j.angle_vel(qp) for j in self.sys.joints]
        
        # Joint angles and velocities
        joint_angles = [jp.array(j[0]).reshape(-1) for j in joint_obs]
        joint_velocities = [jp.array(j[1]).reshape(-1) for j in joint_obs]
        
        # Palm position and orientation
        palm_pos = qp.pos[0]  # palm position
        palm_rot = qp.rot[0]  # palm orientation (quaternion)
        palm_vel = qp.vel[0]  # palm velocity
        palm_ang_vel = qp.ang[0]  # palm angular velocity
        
        # Finger tip positions (approximate - using distal segments)
        finger_tips = qp.pos[3::4][:4] if len(qp.pos) >= 16 else qp.pos[1:5]  # approximate finger tips
        
        # Concatenate all observations
        obs_components = (
            [palm_pos, palm_rot, palm_vel, palm_ang_vel] +
            joint_angles + 
            joint_velocities +
            [finger_tips.reshape(-1)]
        )
        
        return jp.concatenate(obs_components)


# Simplified hand configuration in Brax format
# This is a basic physics setup - in a real implementation, 
# you would need to convert the full MuJoCo XML to Brax config
_SIMPLE_HAND_CONFIG = """
bodies {
  name: "palm"
  colliders {
    box {
      halfsize { x: 0.05 y: 0.03 z: 0.01 }
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 0.235
}

bodies {
  name: "if_bs"
  colliders {
    capsule {
      radius: 0.01
      length: 0.03
    }
  }
  inertia { x: 0.1 y: 0.1 z: 0.1 }
  mass: 0.044
}

bodies {
  name: "if_px"
  colliders {
    capsule {
      radius: 0.008
      length: 0.025
    }
  }
  inertia { x: 0.05 y: 0.05 z: 0.05 }
  mass: 0.032
}

bodies {
  name: "if_md"
  colliders {
    capsule {
      radius: 0.007
      length: 0.022
    }
  }
  inertia { x: 0.04 y: 0.04 z: 0.04 }
  mass: 0.037
}

bodies {
  name: "if_ds"
  colliders {
    capsule {
      radius: 0.006
      length: 0.018
    }
  }
  inertia { x: 0.03 y: 0.03 z: 0.03 }
  mass: 0.016
}

joints {
  name: "if_mcp"
  parent: "palm"
  child: "if_bs"
  parent_offset { x: -0.0825 y: -0.0857 z: 0.0078 }
  child_offset { }
  rotation { z: -90.0 }
  angular_damping: 30.0
  angle_limit { min: -18.0 max: 127.8 }
  stiffness: 5000.0
}

joints {
  name: "if_rot"
  parent: "if_bs"
  child: "if_px"
  parent_offset { x: -0.0122 y: 0.0381 z: 0.0145 }
  child_offset { }
  rotation { z: -90.0 }
  angular_damping: 30.0
  angle_limit { min: -60.0 max: 60.0 }
  stiffness: 5000.0
}

joints {
  name: "if_pip"
  parent: "if_px"
  child: "if_md"
  parent_offset { x: 0.015 y: 0.0143 z: -0.013 }
  child_offset { }
  rotation { z: -90.0 }
  angular_damping: 30.0
  angle_limit { min: -29.0 max: 108.0 }
  stiffness: 5000.0
}

joints {
  name: "if_dip"
  parent: "if_md"
  child: "if_ds"
  parent_offset { x: 0 y: -0.0361 z: 0.0002 }
  child_offset { }
  rotation { z: -90.0 }
  angular_damping: 30.0
  angle_limit { min: -21.0 max: 117.0 }
  stiffness: 5000.0
}

actuators {
  name: "if_mcp_act"
  joint: "if_mcp"
  strength: 300.0
  torque { }
}

actuators {
  name: "if_rot_act"
  joint: "if_rot"
  strength: 300.0
  torque { }
}

actuators {
  name: "if_pip_act"
  joint: "if_pip"
  strength: 300.0
  torque { }
}

actuators {
  name: "if_dip_act"
  joint: "if_dip"
  strength: 300.0
  torque { }
}

forces {
  name: "gravity"
  body: "palm"
  strength: 1.0
  thruster { }
}

friction: 1.0
gravity { z: -9.81 }
angular_damping: -0.05
baumgarte_erp: 0.1
dt: 0.02
substeps: 4
dynamics_mode: "legacy_spring"
"""
