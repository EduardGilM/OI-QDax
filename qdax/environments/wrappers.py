from typing import Dict, Optional

import flax.struct
import jax
import jax.numpy as jnp
from brax.v1 import jumpy as jp
from brax.v1.envs import Env, State, Wrapper
from qdax.environments.lz76 import LZ76, LZ76_jax, action_to_binary_padded


class CompletedEvalMetrics(flax.struct.PyTreeNode):
    current_episode_metrics: Dict[str, jp.ndarray]
    completed_episodes_metrics: Dict[str, jp.ndarray]
    completed_episodes: jp.ndarray
    completed_episodes_steps: jp.ndarray


class CompletedEvalWrapper(Wrapper):
    """Brax env with eval metrics for completed episodes."""

    STATE_INFO_KEY = "completed_eval_metrics"

    def reset(self, rng: jp.ndarray) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=jax.tree_util.tree_map(
                jp.zeros_like, reset_state.metrics
            ),
            completed_episodes_metrics=jax.tree_util.tree_map(
                lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics
            ),
            completed_episodes=jp.zeros(()),
            completed_episodes_steps=jp.zeros(()),
        )
        reset_state.info[self.STATE_INFO_KEY] = eval_metrics
        return reset_state

    def step(self, state: State, action: jp.ndarray) -> State:
        state_metrics = state.info[self.STATE_INFO_KEY]
        if not isinstance(state_metrics, CompletedEvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info[self.STATE_INFO_KEY]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        # steps stores the highest step reached when done = True, and then
        # the next steps becomes action_repeat
        completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
            nstate.info["steps"] * nstate.done
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b, state_metrics.current_episode_metrics, nstate.metrics
        )
        completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
        completed_episodes_metrics = jax.tree_util.tree_map(
            lambda a, b: a + jp.sum(b * nstate.done),
            state_metrics.completed_episodes_metrics,
            current_episode_metrics,
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            current_episode_metrics,
            nstate.metrics,
        )

        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=current_episode_metrics,
            completed_episodes_metrics=completed_episodes_metrics,
            completed_episodes=completed_episodes,
            completed_episodes_steps=completed_episodes_steps,
        )
        nstate.info[self.STATE_INFO_KEY] = eval_metrics
        return nstate


class ClipRewardWrapper(Wrapper):
    """Wraps gym environments to clip the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(
        self,
        env: Env,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        super().__init__(env)
        self._clip_min = clip_min
        self._clip_max = clip_max

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(
            reward=jp.clip(state.reward, a_min=self._clip_min, a_max=self._clip_max)
        )

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(
            reward=jp.clip(state.reward, a_min=self._clip_min, a_max=self._clip_max)
        )


class OffsetRewardWrapper(Wrapper):
    """Wraps gym environments to offset the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(self, env: Env, offset: float = 0.0) -> None:
        super().__init__(env)
        self._offset = offset

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(reward=state.reward + self._offset)

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(reward=state.reward + self._offset)

class LZ76Wrapper(Wrapper):
    """Wraps gym environments to add both Lempel-Ziv complexity and O-Information of the actions taken."""

    def __init__(self, env: Env, max_sequence_length: int = 1000, **kwargs):
        super().__init__(env)
        self.max_sequence_length = max_sequence_length
        self.max_action_binary_length = env.action_size * 32

        # Usar diferentes tamaños de ventana para cada métrica
        self.lz76_window = kwargs.get('lz76_window', self.max_sequence_length)
        self.oi_window = kwargs.get('oi_window', 10)
        
        # Semilla para inicialización
        self.init_key = jax.random.PRNGKey(42)
        
        # Guardar valores mínimos y máximos para normalización
        self.lz76_min = jnp.array(0.0, dtype=jnp.float32)
        self.lz76_max = jnp.array(100.0, dtype=jnp.float32)  # Valor esperado máximo
        self.oi_min = jnp.array(-1.0, dtype=jnp.float32)  # O-Info puede ser negativa
        self.oi_max = jnp.array(1.0, dtype=jnp.float32)
        
    @property
    def action_size(self):
        return self.env.action_size
        
    @property
    def behavior_descriptor_length(self):
        return 2 

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)

        # Inicializar secuencias para LZ76 y O-Info con diferentes patrones
        self.init_key, subkey1, subkey2 = jax.random.split(self.init_key, 3)
        
        # Para LZ76: secuencia binaria inicializada a ceros
        action_sequence = jnp.zeros(
            (self.max_sequence_length, self.max_action_binary_length), dtype=jnp.uint8
        )
        
        # Para O-Info: secuencia de acciones inicializada con pequeñas diferencias
        action_sequence_raw = jnp.zeros(
            (self.max_sequence_length, self.action_size), dtype=jnp.float32
        )
        
        # Guardar secuencias en el estado
        state.info["action_sequence"] = action_sequence
        state.info["action_sequence_raw"] = action_sequence_raw
        
        # Inicializar con valores diferentes para asegurar que no coincidan
        state.info["lz76_complexity"] = jnp.array(0.2, dtype=jnp.float32)
        state.info["o_info_value"] = jnp.array(0.1, dtype=jnp.float32)
        
        # Descriptor de estado con valores diferentes
        state.info["state_descriptor"] = jnp.array([
            state.info["lz76_complexity"], 
            state.info["o_info_value"]
        ])
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)

        current_step = jnp.int32(state.info["steps"] - 1)
        
        # PROCESAMIENTO DE LZ76
        action_binary, _ = action_to_binary_padded(
            action, self.max_action_binary_length
        )
        
        # Actualizar secuencia para LZ76
        action_sequence = state.info["action_sequence"]
        action_sequence = action_sequence.at[current_step % self.max_sequence_length].set(action_binary)
        
        # PROCESAMIENTO DE O-INFORMATION
        # Actualizar secuencia para O-Info 
        # Proyectar acciones al intervalo [0, 1] para que la varianza sea más estable
        normalized_action = (action + 1.0) / 2.0  
        action_sequence_raw = state.info["action_sequence_raw"]
        action_sequence_raw = action_sequence_raw.at[current_step % self.max_sequence_length].set(normalized_action)
        
        # CALCULAR LZ76
        # Considerar solo una parte de la secuencia para controlar su crecimiento
        lz76_end = jnp.minimum(current_step + 1, self.lz76_window)
        lz76_indices = jnp.arange(self.max_sequence_length) < lz76_end
        lz_sequence = action_sequence * lz76_indices[:, jnp.newaxis]
        
        # Aplanar y calcular complejidad
        flattened_sequence = lz_sequence.reshape(-1)
        raw_complexity = jnp.float32(LZ76_jax(flattened_sequence))
        
        # Normalizar LZ76 a un rango [0, 1] para que sea comparable con O-Info
        # Usar un factor dinámico que depende del número de pasos
        lz76_div_factor = jnp.maximum(50.0, current_step / 2.0)  # Evita divisiones por números pequeños
        normalized_complexity = raw_complexity / lz76_div_factor
        normalized_complexity = jnp.minimum(normalized_complexity, 1.0)  # Limitar máximo a 1.0
        
        # CALCULAR O-INFORMATION
        # Usar ventana deslizante para O-Info (últimos oi_window pasos)
        valid_steps = jnp.minimum(current_step + 1, self.max_sequence_length)
        recent_steps = jnp.minimum(valid_steps, self.oi_window)
        
        # Calcular máscara para seleccionar solo los pasos recientes
        recency_mask = jnp.arange(self.max_sequence_length) >= (valid_steps - recent_steps)
        recency_mask = recency_mask & (jnp.arange(self.max_sequence_length) < valid_steps)
        
        # Calcular O-Info y escalar para que no coincida con LZ76
        o_info_raw = jnp.where(
            recent_steps > 1,
            self._compute_o_information(action_sequence_raw, recency_mask),
            jnp.array(0.1, dtype=jnp.float32)
        )
        
        # Escalar para que O-Info esté en un rango diferente a LZ76
        # LZ76 está normalizado a [0, 1], así que normalizamos O-Info a [-0.5, 0.5]
        o_info_norm = o_info_raw * 0.5
        
        # GUARDAR RESULTADOS
        state.info["action_sequence"] = action_sequence
        state.info["action_sequence_raw"] = action_sequence_raw
        state.info["lz76_complexity"] = normalized_complexity
        state.info["o_info_value"] = o_info_norm
        
        # Asegurar que los valores en state_descriptor son diferentes y están en rangos adecuados
        state.info["state_descriptor"] = jnp.array([
            normalized_complexity,  # LZ76 en rango [0, 1]
            o_info_norm            # O-Info en rango [-0.5, 0.5]
        ])
        
        return state
    
    def _compute_o_information(self, action_sequence, valid_mask):
        """Calcula la O-Information utilizando solo acciones recientes"""
        mask = valid_mask[:, jnp.newaxis]
        
        masked_actions = action_sequence * mask

        joint_values = masked_actions.reshape(-1)

        joint_entropy = jnp.log(jnp.var(joint_values) + 1e-8)

        individual_entropies = jnp.array([
            jnp.log(jnp.var(masked_actions[:, i]) + 1e-8)
            for i in range(self.action_size)
        ])
        sum_individual_entropies = jnp.sum(individual_entropies)
        
        total_mutual_info = sum_individual_entropies - joint_entropy

        pair_mutual_info = 0.0
        for i in range(self.action_size):
            for j in range(i + 1, self.action_size): 
                xi = masked_actions[:, i]
                xj = masked_actions[:, j]
                
                pair_data = jnp.stack([xi, xj], axis=-1).reshape(-1)
                pair_joint_entropy = jnp.log(jnp.var(pair_data) + 1e-8)
                
                dimension_weight = 1.0 + 0.1 * (i + j) / (2 * self.action_size)
                
                pair_mi = (individual_entropies[i] + individual_entropies[j] - pair_joint_entropy) * dimension_weight
                pair_mutual_info += pair_mi

        o_info = pair_mutual_info - total_mutual_info
        
        return jnp.tanh(o_info)
