from enum import IntEnum
import jax
import jax.numpy as jnp
from typing import Optional
from nicegui import ui

from flax import struct
import housemaze
import renderer 
import utils
import numpy as np

import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger

'''
python -m ipdb -c continue dyn.py
'''

logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 5000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1

########################################
# Define actions and corresponding keys
########################################

class KeyboardActions(IntEnum):
  right = 0
  down = 1
  left = 2
  up = 3
  done = 4

actions = [KeyboardActions.right, KeyboardActions.down, KeyboardActions.left, KeyboardActions.up, KeyboardActions.done]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Enter"]
action_to_name = [a.name for a in actions]


########################################
# Create environment instance

@struct.dataclass
class MapInit:
  grid: jax.Array
  agent_pos: jax.Array
  agent_dir: jax.Array
  spawn_locs: Optional[jax.Array] = None


image_dict = utils.load_image_dict()

maze2 = """
.#.C...##....
.#..D...####.
.######......
......######.
.#.#..#......
.#.#.##..#...
##.#.#>.###.#
A..#.##..#...
.B.#.........
#####.#..####
......####.#.
.######E.#.#.
........F#...
""".strip()

char_to_key = dict(
  A="knife",
  B="fork",
  C="pan",
  D="pot",
  E="bowl",
  F="plates",
)

object_to_index = {key: idx for idx, key in enumerate(image_dict["keys"])}

objects = np.array([object_to_index[v] for v in char_to_key.values()])

map2_init = utils.from_str(
  maze2, char_to_key=char_to_key, object_to_index=object_to_index
)


seed = 6
rng = jax.random.PRNGKey(seed)

env_params = housemaze.EnvParams(
  map_init=MapInit(*map2_init),
  time_limit=jnp.array(50),
  objects=jnp.asarray(objects),
)

task_runner = housemaze.TaskRunner(task_objects=env_params.objects)
env = housemaze.HouseMaze(
  task_runner=task_runner,
  num_categories=len(image_dict["keys"]),
)
env = utils.AutoResetWrapper(env)
#reset_timestep = env.reset(rng, env_params)

# Create web environment wrapper
jax_web_env = JaxWebEnv(
    env = env,
    actions = action_array,
)

default_params = housemaze.EnvParams(MapInit(*map2_init), objects)

jax_web_env.precompile(dummy_env_params = default_params)


def render_fn(timestep: nicewebrl.TimeStep):
    image = renderer.create_image_from_grid(
    timestep.state.grid,
    timestep.state.agent_pos,
    timestep.state.agent_dir,
    image_dict,
    )
    return image

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
            - Press the arrow keys to move the agent.
            - Press Enter to finish the episode.
            """
        )

instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)

# ------------------
# Environment stage
# ------------------

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
  name="HouseMaze",
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