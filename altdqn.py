import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import random
import collections
from flax.training import train_state
from typing import Tuple, Any, Dict
import matplotlib.pyplot as plt




# Define Q-Network
class QNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x

# Create Training State
class TrainState(train_state.TrainState):
    target_params: Any

def create_train_state(rng, learning_rate, num_actions):
    q_network = QNetwork(num_actions=num_actions)
    params = q_network.init(rng, jnp.ones((1, 5))) 
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=q_network.apply, params=params, target_params=params, tx=tx)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Convert batch to JAX arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            jnp.array(states),
            jnp.array(actions),
            jnp.array(rewards),
            jnp.array(next_states),
            jnp.array(dones)
        )

# Training Step
@jax.jit
def train_step(state, batch, gamma):
    states, actions, rewards, next_states, dones = batch
    def loss_fn(params):
        q_values = state.apply_fn(params, states)
        q_values = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze()
        
        target_q_values = state.apply_fn(state.target_params, next_states)
        max_next_q = jnp.max(target_q_values, axis=1)
        targets = rewards + gamma * max_next_q * (1 - dones)
        loss = jnp.mean((q_values - targets) ** 2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# DQN Training Loop
def train_dqn(env, num_episodes=1000, batch_size=32, gamma=0.99, lr=1e-3, buffer_size=10000, target_update_freq=10):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, lr, env.num_actions)
    buffer = ReplayBuffer(buffer_size)

    rewards_per_episode = [] 
    
    for episode in range(num_episodes):
        obs, env_state = env.reset(rng, env.default_params)
        done = False
        total_reward = 0
        while not done:

            q_values = state.apply_fn(state.params, obs[None, ...])
            action = jnp.argmax(q_values) if random.random() > 0.1 else random.choice([0, 1])
            next_obs, next_state, reward, done, _ = env.step(rng, env_state, action, env.default_params)
            buffer.add((obs, action, reward, next_obs, float(done)))
            obs, env_state = next_obs, next_state
            total_reward += reward
            rewards_per_episode.append(total_reward)
            if len(buffer.buffer) > batch_size:
                batch = buffer.sample(batch_size) 
                state, loss = train_step(state, batch, gamma)
        
        if episode % target_update_freq == 0:
            state = state.replace(target_params=state.params)
        print(f"Episode {episode}, Total Reward: {total_reward}")
    
    plt.plot(rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Reward vs. Time (Episodes)")
    plt.savefig("reward_vs_time.png")
    plt.show()
    
    return state

import jax
import gymnax
from twosteptask import TwoStepTask

# Initialize environment
env = TwoStepTask()
trained_state = train_dqn(env, num_episodes=500)




