"""Example usage of the Left Hand environment."""

import jax
import jax.numpy as jp
from qdax.environments.left_hand import LeftHand


def main():
    """Demonstrate basic usage of the Left Hand environment."""
    
    # Create the environment
    env = LeftHand(
        ctrl_cost_weight=0.01,
        reset_noise_scale=0.1,
        task_reward_weight=1.0,
        backend='generalized'
    )
    
    print(f"Action space size: {env.action_size}")
    print(f"Observation space size: {env.observation_size}")
    print(f"Joint names: {env._joint_names}")
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    
    # Reset the environment
    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    
    print(f"Initial observation shape: {state.obs.shape}")
    print(f"Initial reward: {state.reward}")
    
    # Run a few steps with random actions
    for step in range(10):
        rng, action_rng = jax.random.split(rng)
        
        # Generate random actions (normalized to [-1, 1])
        action = jax.random.uniform(
            action_rng, 
            (env.action_size,), 
            minval=-1.0, 
            maxval=1.0
        )
        
        # Step the environment
        state = env.step(state, action)
        
        print(f"Step {step + 1}:")
        print(f"  Reward: {state.reward:.4f}")
        print(f"  Palm position: [{state.metrics['palm_position_x']:.4f}, "
              f"{state.metrics['palm_position_y']:.4f}, "
              f"{state.metrics['palm_position_z']:.4f}]")
        print(f"  Task reward: {state.metrics['reward_task']:.4f}")
        print(f"  Control cost: {state.metrics['reward_ctrl']:.4f}")
    
    print("\nEnvironment demonstration completed!")


if __name__ == "__main__":
    main()
