import os.path
import asyncio
from asyncio import Lock
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise

import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import stages

import experiment_structure as experiment

DATA_DIR = "data"
DATABASE_FILE = "db.sqlite"

_user_locks = {}


#####################################
# Helper functions
#####################################
# This is used to ensure that each user has a unique lock
def get_user_lock():
  """A function that returns a lock for the current user using their unique seed"""
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


async def experiment_not_finished():
  """Check if the experiment is not finished"""
  async with get_user_lock():
    not_finished = not app.storage.user.get("experiment_finished", False)
    not_finished &= app.storage.user["stage_idx"] < len(experiment.all_stages)
  return not_finished


async def global_handle_key_press(e, container):
  """Define global key press handler

  We can get stage-specific key handling by using this and having this function
  call the stage-specific key handler. When the experiment begins, we'll register
  a key listener to call this function
  """

  stage_idx = app.storage.user["stage_idx"]
  if app.storage.user["stage_idx"] >= len(experiment.all_stages):
    return

  stage = experiment.all_stages[stage_idx]
  if stage.get_user_data("finished", False):
    return

  await stage.handle_key_press(e, container)
  local_handle_key_press = stage.get_user_data("local_handle_key_press")
  if local_handle_key_press is not None:
    await local_handle_key_press()


#####################################
# Setup logger
#####################################
setup_logging(
  DATA_DIR,
  # each user has a unique seed
  # can use this to identify users
  nicegui_storage_user_key="seed",
)
logger = get_logger("main")

#####################################
# Setup database for storing experiment data
#####################################
if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)


async def init_db() -> None:
  await Tortoise.init(
    db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
    # this will use nicewebrl.stages.StageStateModel
    modules={"models": ["nicewebrl.stages"]},
  )
  await Tortoise.generate_schemas()


async def close_db() -> None:
  await Tortoise.close_connections()


app.on_startup(init_db)
app.on_shutdown(close_db)


########################
# Run experiment
########################
async def start_experiment(container):
  # --------------------------------
  # initialize
  # place things like consent form here or getting demographic information here
  # --------------------------------
  if not (app.storage.user.get("experiment_started", False)):
    # NOTE: Here is where you can add a consent form or collect demographic information
    app.storage.user["experiment_started"] = True

  # register global key press handler
  ui.on("key_pressed", lambda e: global_handle_key_press(e, container))

  logger.info("Starting experiment")
  while True and await experiment_not_finished():
    # get current stage
    stage_idx = app.storage.user["stage_idx"]
    stage = experiment.all_stages[stage_idx]

    logger.info("=" * 30)
    logger.info(f"Began stage '{stage.name}'")
    # activate stage
    await run_stage(stage, container)
    logger.info(f"Finished stage '{stage.name}'")

    # wait for any saves to finish before updating stage
    # very important, otherwise may lose data
    if isinstance(stage, stages.EnvStage):
      await stage.finish_saving_user_data()
      logger.info(f"Saved data for stage '{stage.name}'")

    # update stage index
    async with get_user_lock():
      app.storage.user["stage_idx"] = stage_idx + 1

    # check if we've finished all stages
    if app.storage.user["stage_idx"] >= len(experiment.all_stages):
      break

  await finish_experiment(container)


async def finish_experiment(container):
  # NOTE: you can do things like saving data here
  nicewebrl.clear_element(container)
  with container:
    ui.markdown("# Experiment over")


async def run_stage(stage, container):
  """Runs and Environment Stage


  1. create handlers. We provide handlers for key and button presses as an example.
  2. activate the stage
  3. create button if needed

  """
  #########
  # Create an event to signal when the stage is over
  #########
  stage_over_event = asyncio.Event()

  #########
  # create functions for handling key and button presses
  #########
  async def local_handle_key_press():
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage.name} via key press")
        # Signal that the stage is over
        stage_over_event.set()

  await stage.set_user_data(local_handle_key_press=local_handle_key_press)

  async def handle_button_press():
    # check if stage is already finished, if so, return
    if stage.get_user_data("finished", False):
      return

    # handle button press
    await stage.handle_button_press(container)

    # check if stage is finished, if so, signal that the stage is over
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        # Signal that the stage is over
        logger.info(f"Finished {stage.name} via button press")
        stage_over_event.set()

  #############################################
  # Activate new stage
  #############################################
  with container.style("align-items: center;"):
    await stage.activate(container)

  if stage.get_user_data("finished", False):
    # over as soon as stage activation was complete
    logger.info(f"Finished {stage.name} immediately after activation")
    stage_over_event.set()

  if stage.next_button:
    with container:
      button = ui.button("Next page")
      await wait_for_button_or_keypress(button)
      await handle_button_press()

  await stage_over_event.wait()


#####################################
# Root page
#####################################


async def check_if_over(container, episode_limit=60):
  minutes_passed = nicewebrl.get_user_session_minutes()
  minutes_passed = app.storage.user["session_duration"]
  if minutes_passed > episode_limit:
    # define custom behavior on time-out
    pass


@ui.page("/")
async def index(request: Request):
  # collect user information from URL (e.g. from MTurk)
  # can save this
  user_info = dict(
    worker_id=request.query_params.get("workerId", None),
    hit_id=request.query_params.get("hitId", None),
    assignment_id=request.query_params.get("assignmentId", None),
  )

  app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", 0)
  nicewebrl.initialize_user()

  # set up callback to log all pings
  def print_ping(e):
    logger.info(str(e.args))

  ui.on("ping", print_ping)

  ################
  # Start experiment
  ################
  basic_javascript_file = nicewebrl.basic_javascript_file()
  with open(basic_javascript_file) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  card = (
    ui.card(align_items=["center"])
    .classes("fixed-center")
    .style(
      "max-width: 90vw;"  # Set the max width of the card
      "max-height: 90vh;"  # Ensure the max height is 90% of the viewport height
      "overflow: auto;"  # Allow scrolling inside the card if content overflows
      "display: flex;"  # Use flexbox for centering
      "flex-direction: column;"  # Stack content vertically
      "justify-content: flex-start;"
      "align-items: center;"
    )
  )
  with card:
    episode_limit = 200
    with ui.column().style("align-items: center;") as container:
      # check every minute if over
      ui.timer(
        interval=1,
        callback=lambda: check_if_over(
          episode_limit=episode_limit, container=container
        ),
      )
      await start_experiment(container)


ui.run(
  storage_secret="private key to secure the browser session cookie",
  # reload='FLY_ALLOC_ID' not in os.environ,
  reload=True,
  title="HouseMaze Web App",
)