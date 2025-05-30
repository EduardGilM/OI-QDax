"""
Simple script implementing Kaspar & Schuster's algorithm for
Lempel-Ziv complexity (1976 version).

If you use this script, please cite the following paper containing a sample
use case and further description of the use of LZ in neuroscience:

Dolan D. et al (2018). The Improvisational State of Mind: A Multidisciplinary
Study of an Improvisatory Approach to Classical Music Repertoire Performance.
Front. Psychol. 9:1341. doi: 10.3389/fpsyg.2018.01341

Pedro Mediano and Fernando Rosas, 2019
"""
import numpy as np
import jax
import jax.numpy as jnp
from brax.v1 import jumpy as jp

from qdax.custom_types import Descriptor
from qdax.core.neuroevolution.buffers.buffer import QDTransition

def LZ76(ss):
    """
    Calculate Lempel-Ziv's algorithmic complexity using the LZ76 algorithm
    and the sliding-window implementation.

    Reference:

    F. Kaspar, H. G. Schuster, "Easily-calculable measure for the
    complexity of spatiotemporal patterns", Physical Review A, Volume 36,
    Number 2 (1987).

    Input:
      ss -- array of integers

    Output:
      c  -- integer
    """
    ss = ss.flatten().tolist()
    if len(ss) == 0:
        return 0  # Return complexity zero for empty input
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    n = len(ss)
    while True:
        if ss[i + k - 1] == ss[l + k - 1]:
            k = k + 1
            if l + k > n:
                c = c + 1
                break
        else:
            if k > k_max:
               k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c

def action_to_binary(action: jnp.ndarray) -> jnp.ndarray:
    """Convert actions to a JAX-compatible binary representation.
    
    For each floating-point action value, uses 16 bits to represent it
    instead of the full 32 bits for float32 to reduce complexity.
    """
    # Convert to float16 to reduce the bit representation
    action_float16 = action.astype(jnp.float16)
    # Convert float16 to its exact bit representation
    action_view = action_float16.view(jnp.uint16)
    # Unpack bits while preserving the exact binary representation
    binary_rep = jnp.unpackbits(
        action_view.view(jnp.uint8), bitorder='big', axis=-1
    )
    binary_rep_flat = binary_rep.reshape(-1)
    return binary_rep_flat

def action_to_binary_padded(action: jp.ndarray) -> jnp.ndarray:
    """Convierte acciones en una representación binaria de longitud fija.
    
    Args:
        action: Array de acciones a convertir
        max_length: Longitud máxima de la representación binaria
        
    Returns:
        Tupla de (representación binaria con padding, longitud real)
    """
    action = action.flatten()
    
    return action_to_binary(action)

def LZ76_jax(ss: jnp.ndarray) -> jnp.int32:
    """JAX-compatible implementation of the LZ76 algorithm."""
    n = ss.size
    if n == 0:
        return jnp.int32(0)

    def cond_fun(state):
        i, k, l, k_max, c = state
        return (l + k) <= n

    def body_fun(state):
        i, k, l, k_max, c = state
        same = ss[i + k - 1] == ss[l + k - 1]

        def true_branch(_):
            return i, k + 1, l, k_max, c

        def false_branch(_):
            k_max_updated = jax.lax.max(k_max, k)
            i_updated = i + 1

            def inner_true(_):
                c_updated = c + 1
                l_updated = l + k_max_updated
                return 0, 1, l_updated, 1, c_updated

            i_eq_l = i_updated == l
            i_new, k_new, l_new, k_max_new, c_new = jax.lax.cond(
                i_eq_l,
                inner_true,
                lambda _: (i_updated, 1, l, k_max_updated, c),
                operand=None
            )
            return i_new, k_new, l_new, k_max_new, c_new

        i_new, k_new, l_new, k_max_new, c_new = jax.lax.cond(
            same,
            true_branch,
            false_branch,
            operand=None
        )
        return i_new, k_new, l_new, k_max_new, c_new

    state = (0, 1, 1, 1, 1)
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    _, _, _, _, c = state
    return c
