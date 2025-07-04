from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Descriptor, Params

from qdax.environments.lz76 import LZ76, LZ76_jax, action_to_binary


def get_final_xy_position(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Compute final xy positon.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
    descriptors = jax.vmap(lambda x, y: x[y])(data.state_desc, last_index)

    # remove the dim coming from the trajectory
    return descriptors.squeeze(axis=1)


def get_feet_contact_proportion(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Compute feet contact time proportion.

    This function suppose that state descriptor is the feet contact, as it
    just computes the mean of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    descriptors = jnp.sum(data.state_desc * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.sum(1.0 - mask, axis=1)

    return descriptors


def get_lz76_complexity(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Calcula la complejidad de Lempel-Ziv y la O-Information de las acciones tomadas."""
    #jax.debug.print("Data: {x}", x=data.state_desc)
    #jax.debug.print("Data masked: {x}", x=data.state_desc[:, len(data.state_desc[0]) - 1])
    return data.state_desc[:, len(data.state_desc[0]) - 1]


class AuroraExtraInfo(flax.struct.PyTreeNode):
    """
    Information specific to the AURORA algorithm.

    Args:
        model_params: the parameters of the dimensionality reduction model
    """

    model_params: Params


class AuroraExtraInfoNormalization(AuroraExtraInfo):
    """
    Information specific to the AURORA algorithm. In particular, it contains
    the normalization parameters for the observations.

    Args:
        model_params: the parameters of the dimensionality reduction model
        mean_observations: the mean of observations
        std_observations: the std of observations
    """

    mean_observations: jnp.ndarray
    std_observations: jnp.ndarray

    @classmethod
    def create(
        cls,
        model_params: Params,
        mean_observations: jnp.ndarray,
        std_observations: jnp.ndarray,
    ) -> AuroraExtraInfoNormalization:
        return cls(
            model_params=model_params,
            mean_observations=mean_observations,
            std_observations=std_observations,
        )


def get_aurora_encoding(
    observations: jnp.ndarray,
    aurora_extra_info: AuroraExtraInfoNormalization,
    model: flax.linen.Module,
) -> Descriptor:
    """
    Compute final aurora embedding.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    model_params = aurora_extra_info.model_params
    mean_observations = aurora_extra_info.mean_observations
    std_observations = aurora_extra_info.std_observations

    # lstm seq2seq
    normalized_observations = (observations - mean_observations) / std_observations
    descriptors = model.apply(
        {"params": model_params}, normalized_observations, method=model.encode
    )

    return descriptors.squeeze()


def get_hand_final_configuration(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Extract final hand configuration as behavior descriptor.

    This function extracts the final joint positions of the hand fingers
    to characterize the final pose/grasp configuration.

    Args:
        data: QD transition data containing state descriptors
        mask: Mask for valid timesteps

    Returns:
        Descriptor containing final joint positions of key hand segments
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor - final joint configurations
    last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
    descriptors = jax.vmap(lambda x, y: x[y])(data.state_desc, last_index)

    # For the hand, we'll use the first few dimensions which should represent
    # key joint positions (e.g., finger tip positions or key joint angles)
    # This assumes state_desc contains hand joint information
    hand_config = descriptors[:, :2]  # Use first 2 dimensions as behavior descriptor

    return hand_config


def get_hand_movement_diversity(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Extract hand movement diversity as behavior descriptor.

    This function computes the variance/range of hand movements during the episode
    to characterize movement diversity.

    Args:
        data: QD transition data containing state descriptors
        mask: Mask for valid timesteps

    Returns:
        Descriptor containing movement diversity metrics
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get valid state descriptors
    valid_states = data.state_desc * (1.0 - mask)

    # Compute movement diversity metrics
    # Range of movement for key joints
    max_positions = jnp.max(valid_states, axis=1)
    min_positions = jnp.min(valid_states, axis=1)
    movement_range = max_positions - min_positions

    # Use first 2 dimensions of movement range as behavior descriptor
    diversity_descriptor = movement_range[:, :2]

    return diversity_descriptor
