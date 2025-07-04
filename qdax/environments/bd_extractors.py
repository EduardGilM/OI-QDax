from __future__ import annotations

from typing import Any

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import Descriptor, Params


def get_hand_joint_positions(data: Any, mask: Any) -> Any:
    """Get final joint positions for the left hand.

    Args:
        data: environment data containing joint positions
        mask: boolean mask for valid time steps

    Returns:
        Joint positions of the first 4 joints (index finger) as behavior descriptor
    """
    # Extract joint positions (first 4 joints - index finger)
    joint_positions = data.obs[:, :4]  # First 4 joints

    # Get final positions (last valid timestep)
    final_positions = joint_positions[mask.sum(axis=-1) - 1]

    return final_positions


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
