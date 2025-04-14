from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, lax, pmap
import numpy as np
import time

# Parámetros
WINDOW_SIZE = 8192
MIN_MATCH_LEN = 3
MAX_MATCH_LEN = 320
LOOKAHEAD_SIZE = 512
NUM_DEVICES = jax.device_count()

# =====================
# Funciones LZSS
# =====================
@jit
def find_longest_match(data, pos, window_start):
    curr_pos = jnp.minimum(pos, data.shape[0])
    window_end = jnp.minimum(curr_pos, window_start + WINDOW_SIZE)
    lookahead_end = jnp.minimum(curr_pos + LOOKAHEAD_SIZE, data.shape[0])
    
    def body(i, state):
        best_len, best_offset = state
        window_pos = window_start + i
        
        j = jnp.arange(MAX_MATCH_LEN)
        curr_indices = curr_pos + j
        window_indices = window_pos + j
        
        valid = (curr_indices < lookahead_end) & (window_indices < window_end)
        matches = data[curr_indices] == data[window_indices]
        matches = jnp.where(valid, matches, False)
        cumulative = jnp.cumprod(matches)
        match_len = jnp.sum(cumulative)
        
        match_len = jnp.where(match_len >= MIN_MATCH_LEN, match_len, 0)
        
        is_better = match_len > best_len
        best_len = jnp.where(is_better, match_len, best_len)
        best_offset = jnp.where(is_better, curr_pos - window_pos, best_offset)
        
        return best_len, best_offset

    return lax.fori_loop(0, WINDOW_SIZE, body, (0, 0))

@jit
def lzss_compress(data):
    max_output_size = data.shape[0] * 2
    output = jnp.zeros((max_output_size, 3), dtype=jnp.int32)
    pos = 0
    window_start = 0
    output_idx = 0

    def cond(state):
        pos, _, output_idx, _ = state
        return (pos < data.shape[0]) & (output_idx < max_output_size)

    def body(state):
        pos, window_start, output_idx, output = state
        match_len, match_offset = find_longest_match(data, pos, window_start)
        
        def encode_ref(_):
            new_output = output.at[output_idx].set(jnp.array([1, match_offset, match_len]))
            new_pos = pos + match_len
            new_win = jnp.maximum(0, new_pos - WINDOW_SIZE)
            return new_pos, new_win, output_idx + 1, new_output
        
        def encode_lit(_):
            new_output = output.at[output_idx].set(jnp.array([0, data[pos], 0]))
            new_pos = pos + 1
            new_win = jnp.maximum(0, new_pos - WINDOW_SIZE)
            return new_pos, new_win, output_idx + 1, new_output
        
        new_pos, new_win, new_idx, new_out = lax.cond(
            match_len >= MIN_MATCH_LEN,
            encode_ref,
            encode_lit,
            operand=None
        )
        
        return new_pos, new_win, new_idx, new_out

    final_state = lax.while_loop(
        cond,
        body,
        (pos, window_start, output_idx, output)
    )
    
    final_pos, final_win, final_idx, final_out = final_state
    return final_out, final_idx

# =====================
# Versión paralelizada
# =====================
@partial(pmap, axis_name='device_axis')
def find_matches_parallel(data_shard, positions, window_starts, chunk_sizes):
    device_id = jax.lax.axis_index('device_axis')
    pos = positions[0]
    window_start = window_starts[0]
    chunk_size = chunk_sizes[0]
    
    start_idx = device_id * chunk_size
    
    window_pos = window_start + start_idx
    j = jnp.arange(MAX_MATCH_LEN)
    curr_indices = pos + j
    
    # Ensure indices are within bounds
    window_indices = window_pos + j
    data_size = data_shard.shape[0]
    valid = (curr_indices < data_size) & (window_indices < data_size)
    
    # Get matches only where indices are valid
    curr_idx_safe = jnp.where(curr_indices < data_size, curr_indices, 0)
    win_idx_safe = jnp.where(window_indices < data_size, window_indices, 0)
    
    matches = jnp.where(valid, 
                      data_shard[curr_idx_safe] == data_shard[win_idx_safe],
                      False)
    cumulative = jnp.cumprod(matches)
    match_len = jnp.sum(cumulative)
    
    return jnp.where(match_len >= MIN_MATCH_LEN, 
                    jnp.array([match_len, pos - window_pos], dtype=jnp.int32),
                    jnp.array([0, 0], dtype=jnp.int32))

@jit
def lzss_compress_parallel(data):
    max_output_size = data.shape[0] * 2
    output = jnp.zeros((max_output_size, 3), dtype=jnp.int32)
    pos = 0
    window_start = 0
    output_idx = 0
    
    # Create sharded data for pmap
    num_devices = jax.device_count()
    data_per_device = data.shape[0] // num_devices
    if data_per_device == 0:
        data_per_device = 1
    padded_data_size = data_per_device * num_devices
    
    # Pad data if necessary to ensure it's divisible by num_devices
    if padded_data_size > data.shape[0]:
        padded_data = jnp.pad(data, (0, padded_data_size - data.shape[0]))
    else:
        padded_data = data
    
    # Reshape to create shards
    sharded_data = jnp.reshape(padded_data, (num_devices, data_per_device))
    
    # Create replicated scalar arrays for pmap
    # We must create arrays of shape (num_devices,) for each scalar value
    positions_array = jnp.array([pos] * num_devices)
    window_starts_array = jnp.array([window_start] * num_devices)
    chunk_sizes_array = jnp.array([WINDOW_SIZE // num_devices] * num_devices)

    def cond(state):
        pos, _, output_idx, _ = state
        return (pos < data.shape[0]) & (output_idx < max_output_size)

    def body(state):
        pos, window_start, output_idx, output = state
        chunk_size = WINDOW_SIZE // num_devices
        
        # Update replicated scalar arrays for current position
        positions_array = jnp.array([pos] * num_devices)
        window_starts_array = jnp.array([window_start] * num_devices)
        chunk_sizes_array = jnp.array([chunk_size] * num_devices)
        
        # Apply pmap to sharded data
        results = find_matches_parallel(sharded_data, positions_array, window_starts_array, chunk_sizes_array)
        
        best_len = jnp.max(results[:, 0])
        best_offset = results[jnp.argmax(results[:, 0]), 1]
        
        def encode_ref(_):
            new_output = output.at[output_idx].set(jnp.array([1, best_offset, best_len]))
            new_pos = pos + best_len
            new_win = jnp.maximum(0, new_pos - WINDOW_SIZE)
            return new_pos, new_win, output_idx + 1, new_output
        
        def encode_lit(_):
            new_output = output.at[output_idx].set(jnp.array([0, data[pos], 0]))
            new_pos = pos + 1
            new_win = jnp.maximum(0, new_pos - WINDOW_SIZE)
            return new_pos, new_win, output_idx + 1, new_output
        
        new_pos, new_win, new_idx, new_out = lax.cond(
            best_len >= MIN_MATCH_LEN,
            encode_ref,
            encode_lit,
            operand=None
        )
        
        return new_pos, new_win, new_idx, new_out

    final_state = lax.while_loop(
        cond,
        body,
        (pos, window_start, output_idx, output)
    )
    
    final_pos, final_win, final_idx, final_out = final_state
    return final_out, final_idx

# =====================
# Implementación LZ76
# =====================
@jit
def LZ76_jax(ss: jnp.ndarray) -> jnp.int32:
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
            k_max_updated = lax.max(k_max, k)
            i_updated = i + 1

            def inner_true(_):
                return 0, 1, l + k_max_updated, 1, c + 1

            i_eq_l = i_updated == l
            i_new, k_new, l_new, k_max_new, c_new = lax.cond(
                i_eq_l,
                inner_true,
                lambda _: (i_updated, 1, l, k_max_updated, c),
                operand=None
            )
            return i_new, k_new, l_new, k_max_new, c_new

        return lax.cond(same, true_branch, false_branch, operand=None)

    state = (0, 1, 1, 1, 1)
    final_state = lax.while_loop(cond_fun, body_fun, state)
    return final_state[4]

# ===============
# Tests de precisión
# ===============
def run_precision_tests():
    test_cases = [
        (jnp.array([0]*100), "100 ceros"),
        (jnp.array([0,1]*50), "01 repetido"),
        (jnp.array(np.random.randint(0,2,1000)), "Aleatorio 1KB"),
        (jnp.array(np.random.randint(0,2,100000)), "Aleatorio 100KB")
    ]
    
    print(f"{'Caso':<20} | {'LZSS Tokens':<12} | {'LZ76 Frases':<12} | {'Ratio LZSS/LZ76'}")
    print("-"*65)
    
    for data, name in test_cases:
        # Compresión LZSS
        compressed_tuple = lzss_compress(data)
        final_out = compressed_tuple[0].block_until_ready()
        final_idx = compressed_tuple[1].block_until_ready()
        compressed = final_out[:final_idx]
        lzss_tokens = compressed.shape[0]
        
        # Medición LZ76
        lz76_phrases = LZ76_jax(data).block_until_ready()
        
        ratio = lzss_tokens / lz76_phrases if lz76_phrases != 0 else 0
        print(f"{name:<20} | {lzss_tokens:<12} | {lz76_phrases:<12} | {ratio:.2f}")

# ===============
# Tests de rendimiento
# ===============
def run_performance_tests():
    data_sizes = [1000, 10000, 100000, 1000000]
    warmup_data = jnp.array(np.random.randint(0, 2, 10000))
    
    # Warmup
    compressed_warmup = lzss_compress(warmup_data)
    compressed_warmup[0].block_until_ready()
    compressed_warmup[1].block_until_ready()
    
    compressed_warmup_parallel = lzss_compress_parallel(warmup_data)
    compressed_warmup_parallel[0].block_until_ready()
    compressed_warmup_parallel[1].block_until_ready()
    
    LZ76_jax(warmup_data).block_until_ready()
    
    print(f"{'Tamaño':<10} | {'LZSS Seq':<10} | {'LZSS Par':<10} | {'LZ76':<10} | Speedup")
    print("-"*55)
    
    for size in data_sizes:
        data = jnp.array(np.random.randint(0, 2, size))
        
        # LZSS secuencial
        start = time.time()
        compressed_seq = lzss_compress(data)
        compressed_seq[0].block_until_ready()
        compressed_seq[1].block_until_ready()
        seq_time = time.time() - start
        
        # LZSS paralelo
        start = time.time()
        compressed_par = lzss_compress_parallel(data)
        compressed_par[0].block_until_ready()
        compressed_par[1].block_until_ready()
        par_time = time.time() - start
        
        # LZ76
        start = time.time()
        LZ76_jax(data).block_until_ready()
        lz76_time = time.time() - start
        
        speedup = seq_time / par_time if par_time > 0 else 0
        print(f"{size:<10} | {seq_time:.4f}s | {par_time:.4f}s | {lz76_time:.4f}s | {speedup:.2f}x")

# ===============
# Ejecución principal
# ===============
if __name__ == "__main__":
    print("========== TEST DE PRECISIÓN ==========")
    run_precision_tests()
    
    print("\n========== TEST DE RENDIMIENTO ==========")
    run_performance_tests()