from typing import Any, Dict, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp


@struct.dataclass
class EnvState:
  step: int
  choice: int
  reward: float

@struct.dataclass
class EnvParams:
  max_steps_in_episode: int = 2

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

    choice = jax.lax.select(state.step == 1, action, state.choice)
    reward = jax.lax.select(state.step == 1, 0.0, jax.random.choice(key, jnp.array([0.0, 1.0]), p=jnp.array([0.5, 0.5])))

    new_state = EnvState(step=step, choice=choice, reward=reward)
    done = step >= params.max_steps_in_episode
    print(f"step: {step}, choice: {choice}, reward: {reward}, done: {done}")
    return self.get_obs(new_state), new_state, reward, done, {}

  def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
    """Reset environment state."""
    state = EnvState(step=0, choice=-1, reward=0.0)
    return self.get_obs(state), state

  def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
    """Return observation from raw state trafo."""
    return jnp.array([state.step, state.choice, state.reward])

  def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Check whether state is terminal."""
    return state.step >= params.max_steps_in_episode

  def discount(self, state, params) -> jnp.ndarray:
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

    choice_is_minus_one = jnp.equal(state.choice, -1)
    choice_is_zero = jnp.equal(state.choice, 0)
    choice_is_one = jnp.equal(state.choice, 1)

    canvas = jax.lax.cond(
        choice_is_zero,
        lambda canvas: canvas.at[1, 0].set(1.0).at[0, 0].set(1.0).at[2, 0].set(1.0),
        lambda canvas: canvas,
        canvas
    )
    canvas = jax.lax.cond(
        choice_is_one,
        lambda canvas: canvas.at[1, 2].set(1.0).at[0, 2].set(1.0).at[2, 2].set(1.0),
        lambda canvas: canvas,
        canvas
    )

    return canvas


