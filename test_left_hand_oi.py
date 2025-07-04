#!/usr/bin/env python3
"""Test the left_hand_oi environment (with LZ76 complexity)."""

import jax
import jax.numpy as jnp
from qdax.environments import create

def main():
    """Test the left_hand_oi environment."""
    
    print("Testing left_hand_oi environment...")
    
    # Create the environment
    env = create("left_hand_oi", episode_length=100, batch_size=1)
    
    print(f"Left hand OI environment created successfully!")
    print(f"Action space size: {env.action_size}")
    print(f"Observation space size: {env.observation_size}")
    
    # Test basic functionality
    key = jax.random.PRNGKey(42)
    state = env.reset(key)
    
    print(f"Initial state shape: {state.obs.shape}")
    print(f"Initial reward: {state.reward}")
    print(f"Initial done: {state.done}")
    
    # Take a random action
    action = jax.random.uniform(key, (1, env.action_size), minval=-1.0, maxval=1.0)
    new_state = env.step(state, action)
    
    print(f"After step - reward: {new_state.reward}, done: {new_state.done}")
    
    print("Left hand OI environment is working correctly!")

if __name__ == "__main__":
    main()
