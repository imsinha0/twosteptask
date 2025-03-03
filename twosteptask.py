from typing import Any, Dict, Tuple
import chex
from flax import struct
import jax
import jax.numpy as jnp
import gymnax
from gymnax.environments import environment, spaces

@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 2  # Default: Two-Step Task
    images: jnp.ndarray = jnp.array([
        # State 0 (Start): Cross pattern
        [[0.0, 1.0, 0.0],
         [1.0, 1.0, 1.0],
         [0.0, 1.0, 0.0]],
        # State 1 (Left): Left arrow or skew
        [[1.0, 0.0, 0.0],
         [1.0, 1.0, 0.0],
         [1.0, 0.0, 0.0]],
        # State 2 (Right): Right arrow or skew
        [[0.0, 0.0, 1.0],
         [0.0, 1.0, 1.0],
         [0.0, 0.0, 1.0]]
    ])  # Shape: (3, 3, 3) - 3 states, each 3x3
    reward_probs: jnp.ndarray = jnp.array([
        [0.0, 1.0, 0.0],  # Start state (dummy, no reward)
        [0.3, 0.4, 0.3],  # Left state
        [0.6, 0.3, 0.1],  # Right state
    ])  # Shape: (num_states, num_rewards)
    num_states: int = reward_probs.shape[0]

@struct.dataclass
class EnvState:
    step: int
    state_idx: int  # Index of the current state
    reward: float

class TwoStepTask(environment.Environment):
    """JAX Compatible version of a Two-Step Task (or N-Step Task) environment."""

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()  # Defaults to Two-Step Task

    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        step = state.step + 1
        new_state_idx = (state.state_idx + action) % params.num_states
        reward_probs = params.reward_probs[new_state_idx]
        reward = jax.random.choice(key, jnp.array([0.0, 1.0, 2.0]), p=reward_probs)
        new_state = EnvState(step=step, state_idx=new_state_idx, reward=reward)
        done = step >= params.max_steps_in_episode
        info = {"discount": jax.lax.select(done, 0.0, 1.0),
                "returned_episode_returns": reward * (1 - done)}  # Simple logging for metrics
        return self.get_obs(new_state), new_state, reward, done, info

    def reset(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        state = EnvState(step=0, state_idx=0, reward=0.0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state."""
        return jnp.array([state.step, state.state_idx, state.reward], dtype=jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        return state.step >= params.max_steps_in_episode

    def discount(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        return "TwoStepTask" if self.default_params.max_steps_in_episode == 2 else "NStepTask"

    @property
    def num_actions(self) -> int:
        return 2  # Actions: 0 (stay/left), 1 (right)

    def action_space(self, params: EnvParams = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array([0, 0, 0.0], dtype=jnp.float32)  # step, state_idx, reward
        high = jnp.array([params.max_steps_in_episode, params.num_states - 1, 2.0], dtype=jnp.float32)
        return spaces.Box(low, high, shape=(3,), dtype=jnp.float32)
    
    @property
    def observation_shape(self) -> tuple:
        return (3,)

def render(state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Return an image-like representation of the current state."""
    image = params.images[state.state_idx]

    return image
