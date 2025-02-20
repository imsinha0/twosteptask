from enum import IntEnum
import jax
import jax.numpy as jnp
from typing import Optional
from nicegui import ui

from flax import struct
import twosteptask

import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger

logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1

########################################
# Define actions and corresponding keys
########################################

class Actions(IntEnum):
    #select action 0, select action 1
    left = 0
    right = 1

actions = [Actions.left, Actions.right]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowLeft", "ArrowRight"]
action_to_name = [a.name for a in actions]

########################################
# Define TwoStep environment
########################################
# Create environment instance
twostep_env = twosteptask.TwoStepTask()

jax_env = TimestepWrapper(
    env = twostep_env,
    autoreset = True,
    use_params = True,
)

# Create web environment wrapper
jax_web_env = JaxWebEnv(
    env = jax_env,
    actions = action_array,
)

default_params = twostep_env.default_params

jax_web_env.precompile(dummy_env_params = default_params)

def render_fn(timestep: nicewebrl.TimeStep):
    return twosteptask.render(timestep.state)

# JIT compile
render_fn = jax.jit(render_fn)

# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(
  render_fn, default_params
)

########################################
# Define Stages of experiment
########################################
all_stages = []

# ------------------
# Instruction stage
# ------------------
async def instruction_display_fn(stage, container):
    with container.style("align-items: center;"):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}")
        ui.markdown(
            """
            - Press "Left Arrow" for Action 0
            - Press "Right Arrow" for Action 1
            """
        )

instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)

# ------------------
# Environment stage
# ------------------

env_params = jax_env.default_params

def make_image_html(src):
  html = f"""
  <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
      <img id="stateImage" src="{src}" style="width: 400px; height: 400px; object-fit: contain;">
  </div>
  """
  return html

async def env_stage_display_fn(
  stage: EnvStage, container: ui.element, timestep: TimeStep
):
  state_image = stage.render_fn(timestep)
  state_image = base64_npimage(state_image)
  stage_state = stage.get_user_data("stage_state")

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    # --------------------------------
    # tell person how many episodes completed and how many successful
    # --------------------------------
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{stage.min_success}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )

    # --------------------------------
    # display environment
    # --------------------------------
    ui.html(make_image_html(src=state_image))

def evaluate_success_fn(timestep: TimeStep, params: Optional[struct.PyTreeNode] = None):
  """Episode finishes if person gets 5 achievements"""
  return timestep.last()# and timestep.reward > 0

environment_stage = EnvStage(
  name="Environment",
  web_env=jax_web_env,
  action_keys=action_keys,
  action_to_name=action_to_name,
  env_params=env_params,
  render_fn=render_fn,
  vmap_render_fn=vmap_render_fn,
  display_fn=env_stage_display_fn,
  evaluate_success_fn=evaluate_success_fn,
  min_success=MIN_SUCCESS_EPISODES,
  max_episodes=MAX_STAGE_EPISODES,
  verbosity=VERBOSITY,
  # add custom metadata to be stored here
  metadata=dict(
    # nothing required, just for bookkeeping
    desc="some description",
    key1="value1",
    key2="value2",
  ),
)
all_stages.append(environment_stage)
