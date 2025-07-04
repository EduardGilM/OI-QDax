#!/usr/bin/env python3
"""Simple example showing how to use the left_hand environment with QDax Map-Elites."""

import jax
import jax.numpy as jnp
from qdax.environments import create
from qdax.environments.bd_extractors import get_hand_final_configuration

def main():
    """Example usage of the left_hand environment."""
    
    print("Creating left_hand environment...")
    
    # Create the environment
    env = create("left_hand", episode_length=100, batch_size=1)
    
    print(f"Environment created successfully!")
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
    
    print("Left hand environment is working correctly!")
    
    # You can now use this environment with QDax algorithms like:
    # - MAP-Elites
    # - CMA-ME
    # - PGA-ME
    # - etc.
    
    print("\nTo use with MAP-Elites, you can do:")
    print("env = create('left_hand', episode_length=1000)")
    print("# Then use with your preferred QDax algorithm")

if __name__ == "__main__":
    main()
