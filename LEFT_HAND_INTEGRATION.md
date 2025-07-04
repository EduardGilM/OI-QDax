# Left Hand Environment for QDax

## Overview

The left_hand environment is a robotic manipulation environment based on the LEAP Hand robot model. It has been successfully integrated into the QDax framework for quality-diversity experiments and can be used with all QDax algorithms like MAP-Elites.

## Features

### Robot Specifications
- **16 joints** representing a left hand with 4 fingers:
  - **Index finger**: 4 joints (mcp, rot, pip, dip)
  - **Middle finger**: 4 joints (mcp, rot, pip, dip) 
  - **Ring finger**: 4 joints (mcp, rot, pip, dip)
  - **Thumb**: 4 joints (cmc, axl, mcp, ipl)

### Environment Properties
- **Action space**: 7 actuated joints (simplified model)
- **Observation space**: 33 dimensions (joint positions, velocities, etc.)
- **Physics**: Brax-based simulation with realistic joint limits and dynamics

### Available Variants
1. **`left_hand`**: Standard left hand environment
2. **`left_hand_oi`**: Left hand with LZ76 complexity wrapper for Open-Ended Innovation

## Usage

### Basic Usage

```python
import jax
import jax.numpy as jnp
from qdax.environments import create

# Create the environment
env = create("left_hand", episode_length=1000, batch_size=1)

# Reset and step
key = jax.random.PRNGKey(42)
state = env.reset(key)

# Random action
action = jax.random.uniform(key, (1, env.action_size), minval=-1.0, maxval=1.0)
new_state = env.step(state, action)

print(f"Action space size: {env.action_size}")
print(f"Observation space size: {env.observation_size}")
```

### With MAP-Elites

```python
from qdax.environments import create
from qdax.core.map_elites import MAPElites

# Create environment and MAP-Elites algorithm
env = create("left_hand", episode_length=1000)
map_elites = MAPElites(
    # ... algorithm parameters
)

# Run evolution
# ... standard MAP-Elites usage
```

### With Open-Ended Innovation

```python
from qdax.environments import create

# Create environment with LZ76 complexity wrapper
env = create("left_hand_oi", episode_length=1000, batch_size=1)

# This environment automatically computes LZ76 complexity as behavior descriptor
```

## Behavior Descriptors

The environment comes with specialized behavior descriptor extractors:

1. **`get_hand_final_configuration`**: Extracts the final joint configuration as a 2D descriptor
2. **`get_hand_movement_diversity`**: Measures the diversity of hand movements throughout the episode
3. **`get_lz76_complexity`**: Computes LZ76 complexity (for `left_hand_oi` variant)

## Environment Details

### Rewards
The environment provides rewards based on:
- Joint movement efficiency
- Reaching target configurations
- Avoiding joint limits

### Behavior Space
- **For `left_hand`**: 2D behavior space based on hand configuration
- **For `left_hand_oi`**: 2D behavior space based on LZ76 complexity

### Physics Parameters
- **Joint stiffness**: 5000.0 (compatible with Brax legacy_spring dynamics)
- **Angular damping**: 30.0 for all joints
- **Joint limits**: Based on real LEAP Hand specifications
- **Timestep**: 0.02 seconds
- **Substeps**: 4

## Files

The integration consists of several files:

1. **`qdax/environments/left_hand.py`**: Main environment class
2. **`qdax/environments/bd_extractors.py`**: Behavior descriptor extractors
3. **`qdax/environments/__init__.py`**: Environment registration
4. **`qdax/environments/wrappers.py`**: LZ76 wrapper support
5. **`assets/left_hand.xml`**: Original MuJoCo XML asset

## Example Scripts

### Basic Test
```python
# File: test_left_hand.py
import jax
from qdax.environments import create

env = create("left_hand", episode_length=100)
key = jax.random.PRNGKey(0)
state = env.reset(key)
print(f"Environment created with {env.action_size} actions and {env.observation_size} observations")
```

### MAP-Elites Integration
```python
# File: left_hand_map_elites.py
from qdax.environments import create
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

# Environment setup
env = create("left_hand", episode_length=1000)
centroids = compute_cvt_centroids(
    num_descriptors=env.behavior_descriptor_length,
    num_init_cvt_samples=50000,
    num_centroids=1024,
    minval=0.0,
    maxval=1.0,
)

# MAP-Elites setup
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=emitter,
    metrics_function=metrics_fn,
)

# Run evolution
repertoire, emitter_state, random_key = map_elites.init(
    init_genotypes, centroids, random_key
)
```

## Compatibility

- **QDax version**: Compatible with current QDax framework
- **JAX version**: Tested with JAX 0.6.2 (includes compatibility fixes)
- **Brax version**: Uses Brax v1 API with legacy_spring dynamics

## Known Issues

1. **JAX Version**: The environment includes compatibility fixes for newer JAX versions
2. **LZ76 Normalization**: Uses estimated normalization values for LZ76 complexity
3. **Simplified Physics**: Uses a simplified physics model rather than full MuJoCo conversion

## Next Steps

1. **Calibrate LZ76 values**: Run experiments to determine proper normalization values
2. **Add more behavior descriptors**: Implement additional hand-specific descriptors
3. **Enhance physics model**: Improve the physics simulation for more realistic behavior
4. **Add visualization**: Create visualization tools for hand movements

## Contributing

To extend the left_hand environment:

1. Add new behavior descriptors in `bd_extractors.py`
2. Update environment registration in `__init__.py`
3. Test with example scripts
4. Update documentation

The environment is now fully integrated and ready for quality-diversity experiments!
