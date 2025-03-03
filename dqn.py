# DQN Implementation

import os
import jax
import jax.numpy as jnp
import chex
import flax
import wandb
import optax
import flax.linen as nn
import flashbax as fbx
from flax.training.train_state import TrainState
from typing import Any, Dict, Tuple
import twosteptask

class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    
    # Instantiate the custom environment
    env = twosteptask.TwoStepTask()
    env_params = env.default_params

    # Vectorized reset and step for multiple environments
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def train(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(jax.random.split(_rng, config["NUM_ENVS"]), env_params)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng_dummy = jax.random.PRNGKey(0)
        _obs, _env_state = env.reset(rng_dummy, env_params)
        _action = jax.random.randint(rng_dummy, shape=(), minval=0, maxval=env.num_actions)
        _obs_next, _, _reward, _done, _ = env.step(rng_dummy, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(action_dim=env.num_actions)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_shape)
        network_params = network.init(_rng, init_x)

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.adam(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        # Epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(rng, 2)
            eps = jnp.clip(
                ((config["EPSILON_FINISH"] - config["EPSILON_START"]) / config["EPSILON_ANNEAL_TIME"]) * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape) < eps,
                jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
                greedy_actions,
            )
            return chosed_actions

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs)
            action = eps_greedy_exploration(rng_a, q_vals, train_state.timesteps)
            obs, env_state, reward, done, info = vmap_step(
                jax.random.split(rng_s, config["NUM_ENVS"]), env_state, action, env_params
            )
            train_state = train_state.replace(timesteps=train_state.timesteps + config["NUM_ENVS"])

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):
                learn_batch = buffer.sample(buffer_state, rng).experience
                q_next_target = network.apply(train_state.target_network_params, learn_batch.second.obs)
                q_next_target = jnp.max(q_next_target, axis=-1)
                target = learn_batch.first.reward + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target

                def _loss_fn(params):
                    q_vals = network.apply(params, learn_batch.first.obs)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals, jnp.expand_dims(learn_batch.first.action, axis=-1), axis=-1
                    ).squeeze(axis=-1)
                    return jnp.mean((chosen_action_qvals - target) ** 2)

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (train_state.timesteps > config["LEARNING_STARTS"])
                & (train_state.timesteps % config["TRAINING_INTERVAL"] == 0)
            )
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, jnp.array(0.0)),
                train_state,
                _rng,
            )

            # Update target network
            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params, train_state.target_network_params, config["TAU"]
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": jnp.mean(info["returned_episode_returns"]),
            }

            if config.get("WANDB_MODE", "disabled") == "online":
                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)
                jax.debug.callback(callback, metrics)

            runner_state = (train_state, buffer_state, env_state, obs, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

def main():
    config = {
        "NUM_ENVS": 10,
        "BUFFER_SIZE": 10000,
        "BUFFER_BATCH_SIZE": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 10000,
        "TRAINING_INTERVAL": 10,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "online",
        "ENTITY": "imsinha-harvard-university",
        "PROJECT": "TwoStepTask",
    }

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["DQN", "TWOSTEPTASK", f"jax_{jax.__version__}"],
        name="purejaxrl_dqn_twosteptask",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

if __name__ == "__main__":
    main()