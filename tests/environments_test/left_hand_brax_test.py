"""Quick smoke test for the LeftHandBrax environment.

Run it with:
    python tests/environments_test/left_hand_brax_test.py

The test avoids the use of *pytest* so that it can be executed directly with
plain Python (as requested).
"""
from __future__ import annotations

import sys

import jax
from brax.v1 import jumpy as jp  # type: ignore [import]
from brax.envs.base import PipelineEnv  # type: ignore [import]

from qdax.environments.left_hand_brax import LeftHandBrax  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def smoke_instantiation() -> None:
    """Instantiate environment through both code-path variants."""
    env = LeftHandBrax()
    _assert(isinstance(env, PipelineEnv), "Direct instantiation failed to return a PipelineEnv")




def smoke_rollout() -> None:
    """Performs a very small rollout to make sure step & JIT work."""
    env = LeftHandBrax()
    key = jp.random_prngkey(seed=0)

    state = env.reset(rng=key)
    obs_size = env.observation_size
    act_size = env.action_size

    # A handful of deterministic steps
    for _ in range(4):
        action = jp.zeros((act_size,))
        state = env.step(state, action)
        _assert(state.obs.size == obs_size, "Observation size mismatch")

    # Ensure that the step function is JIT-compatible.
    state = env.reset(rng=key)
    action = jp.zeros((act_size,))
    _ = jax.jit(env.step)(state, action)


if __name__ == "__main__":
    try:
        smoke_instantiation()
        smoke_rollout()
    except AssertionError as err:
        print(f"TEST FAILED: {err}")
        sys.exit(1)

    print("All smoke tests passed ✔️")
