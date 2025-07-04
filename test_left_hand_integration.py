#!/usr/bin/env python3
"""Test script to validate the left_hand environment integration with QDax."""

import jax
import jax.numpy as jnp
from qdax.environments import create
from qdax.environments.bd_extractors import get_hand_final_configuration

def test_left_hand_environment():
    """Test that the left_hand environment can be created and used."""
    
    print("Testing left_hand environment creation...")
    
    # Create the left_hand environment
    try:
        env = create("left_hand", episode_length=100, batch_size=1)
        print("✓ Left hand environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create left_hand environment: {e}")
        return False
    
    # Test environment reset
    try:
        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        print("✓ Environment reset successful")
        print(f"  - State shape: {state.obs.shape}")
        print(f"  - Action space: {env.action_size}")
        print(f"  - Observation space: {env.observation_size}")
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        return False
    
    # Test a single step
    try:
        action = jnp.zeros((1, env.action_size))
        new_state = env.step(state, action)
        print("✓ Environment step successful")
        print(f"  - New state shape: {new_state.obs.shape}")
        print(f"  - Reward: {new_state.reward}")
    except Exception as e:
        print(f"✗ Failed to step environment: {e}")
        return False
    
    return True

def test_left_hand_oi_environment():
    """Test that the left_hand_oi environment can be created and used."""
    
    print("\nTesting left_hand_oi environment creation...")
    
    # Create the left_hand_oi environment
    try:
        env = create("left_hand_oi", episode_length=100, batch_size=1)
        print("✓ Left hand OI environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create left_hand_oi environment: {e}")
        return False
    
    # Test environment reset
    try:
        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        print("✓ OI Environment reset successful")
        print(f"  - State shape: {state.obs.shape}")
    except Exception as e:
        print(f"✗ Failed to reset OI environment: {e}")
        return False
    
    return True

def test_behavior_descriptor_extractor():
    """Test that the behavior descriptor extractor works."""
    
    print("\nTesting behavior descriptor extractor...")
    
    try:
        # Create environment
        env = create("left_hand", episode_length=10, batch_size=1)
        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        
        # Run a few steps to get some data
        actions = jnp.zeros((10, 1, env.action_size))
        states = []
        
        for i in range(10):
            states.append(state)
            action = actions[i]
            state = env.step(state, action)
        
        # Create a mock QDTransition-like object
        class MockTransition:
            def __init__(self, obs, rewards, dones):
                self.obs = obs
                self.rewards = rewards
                self.dones = dones
        
        # Stack observations, rewards, and dones
        obs = jnp.stack([s.obs for s in states])
        rewards = jnp.stack([s.reward for s in states])
        dones = jnp.stack([s.done for s in states])
        
        transition = MockTransition(obs, rewards, dones)
        mask = jnp.ones_like(dones)
        
        # Test the behavior descriptor extractor
        descriptor = get_hand_final_configuration(transition, mask)
        print(f"✓ Behavior descriptor extracted successfully")
        print(f"  - Descriptor shape: {descriptor.shape}")
        print(f"  - Descriptor values: {descriptor}")
        
    except Exception as e:
        print(f"✗ Failed to extract behavior descriptor: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Left Hand Environment Integration Test")
    print("=" * 50)
    
    success = True
    success &= test_left_hand_environment()
    success &= test_left_hand_oi_environment()
    success &= test_behavior_descriptor_extractor()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Left hand environment is ready to use.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 50)
