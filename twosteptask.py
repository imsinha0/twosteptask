from typing import Any, Dict, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp

'''
@struct.dataclass
class EnvState:
    step: int
    state_idx: int  # 0: start, 1: left, 2: right
    reward: float

@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 2
    images: jnp.ndarray = jnp.zeros((3, 3, 3))
    left_reward_probs: jnp.ndarray = jnp.array([0.3, 0.4, 0.3])
    right_reward_probs: jnp.ndarray = jnp.array([0.6, 0.3, 0.1])
    center_reward_values: jnp.ndarray = jnp.array([0.0, 1.0, 2.0])

'''

@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 2
    images: jnp.ndarray = jnp.zeros((3, 3, 3))
    left_reward_probs: jnp.ndarray = jnp.array([0.3, 0.4, 0.3])
    right_reward_probs: jnp.ndarray = jnp.array([0.6, 0.3, 0.1])
    center_reward_values: jnp.ndarray = jnp.array([0.0, 1.0, 2.0])

@struct.dataclass
class EnvState:
    step: int
    state_idx: int  # 0: start, 1: left, 2: right
    reward: float

@struct.dataclass
class State:
    image: jnp.ndarray
    rewards: jnp.ndarray


@struct.dataclass
class Domain:
    start: State
    left: State
    right: State


class TwoStepTask:
    """JAX Compatible version of a Two-Step Task environment."""
    
    def __init__(self):
        super().__init__()
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        step = state.step + 1
        reward = 0.0
        new_state_idx = action + 1  # 0 -> left (1), 1 -> right (2)
        
        reward_probs = jax.lax.select(
            new_state_idx == 1, params.left_reward_probs, params.right_reward_probs
        )
        reward = jax.random.choice(key, jnp.array([0.0, 1.0, 2.0]), p=reward_probs)
        
        new_state = EnvState(step=step, state_idx=new_state_idx, reward=reward)
        done = step >= params.max_steps_in_episode
        
        return self.get_obs(new_state, params), new_state, reward, done, {}
    
    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Reset environment state."""
        state = EnvState(step=0, state_idx=0, reward=0.0)
        return self.get_obs(state, params), state
    
    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return observation from raw state."""
        return jnp.array([
            state.step,
            state.state_idx,
            state.reward
        ])
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        return state.step >= params.max_steps_in_episode
    
    def discount(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)
    
    @property
    def name(self) -> str:
        return "TwoStepTask"
    
    @property
    def num_actions(self) -> int:
        return 2

def render(state: EnvState):
    """Return a string representation of the game state."""
    canvas = jnp.zeros((3, 3))
    
    canvas = canvas.at[1, 0].set(1.0)
    canvas = canvas.at[1, 2].set(1.0)
    
    state_is_left = jnp.equal(state.state_idx, 1)
    state_is_right = jnp.equal(state.state_idx, 2)
    
    canvas = jax.lax.cond(
        state_is_left,
        lambda canvas: canvas.at[1, 0].set(1.0).at[0, 0].set(1.0).at[2, 0].set(1.0),
        lambda canvas: canvas,
        canvas
    )
    canvas = jax.lax.cond(
        state_is_right,
        lambda canvas: canvas.at[1, 2].set(1.0).at[0, 2].set(1.0).at[2, 2].set(1.0),
        lambda canvas: canvas,
        canvas
    )
    
    return canvas