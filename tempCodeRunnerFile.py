from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from collections import deque
import numpy as np
import optax
import twosteptask

import jax
import jax.numpy as jnp
import flax.linen as nn


# Q-Network definition
class QNetwork(nn.Module):
    hidden_dims: Sequence[int] = (64, 64)
    action_dim: int = 2
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return (jnp.array(states), jnp.array(actions), jnp.array(rewards), 
                jnp.array(next_states), jnp.array(dones))
    
    def __len__(self):
        return len(self.buffer)
from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from collections import deque
import numpy as np
import optax

# ... (keep QNetwork and ReplayBuffer unchanged) ...

class DQNAgent:
    def __init__(self, 
                 env: twosteptask.TwoStepTask,
                 seed: int = 0,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        
        self.env = env
        rng = random.PRNGKey(seed)
        self.rng, init_rng = random.split(rng)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.step_count = 0
        
        # Initialize networks
        self.q_network = QNetwork()
        dummy_obs = jnp.zeros((1, 3))
        self.params = self.q_network.init(init_rng, dummy_obs)
        self.target_params = self.params
        
        # Optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        self.target_update_freq = target_update_freq
    
    def get_action(self, state: jnp.ndarray, eval: bool = False) -> int:
        if not eval and random.uniform(self.rng) < self.epsilon:
            self.rng, _ = random.split(self.rng)
            return random.randint(self.rng, (), 0, self.env.num_actions)
        
        state = jnp.expand_dims(state, 0)
        q_values = self.q_network.apply(self.params, state)
        return jnp.argmax(q_values).item()
    
    def compute_loss(self, params, target_params, batch, q_network):
        states, actions, rewards, next_states, dones = batch
        
        # Current Q values
        q_values = q_network.apply(params, states)
        q_values = q_values[jnp.arange(len(actions)), actions]
        
        # Target Q values
        next_q_values = q_network.apply(target_params, next_states)
        next_q_values = jnp.max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        return jnp.mean((q_values - targets) ** 2)
    
    def update_step(self, params, target_params, opt_state, batch, q_network):
        loss, grads = jax.value_and_grad(self.compute_loss)(params, target_params, batch, q_network)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    def train(self, num_episodes: int):
        total_rewards = []
        
        for episode in range(num_episodes):
            self.rng, key = random.split(self.rng)
            obs, state = self.env.reset(key)
            episode_reward = 0
            done = False
            
            while not done:
                action = self.get_action(obs)
                self.rng, key = random.split(self.rng)
                next_obs, next_state, reward, done, _ = self.env.step(
                    key, state, action, self.env.default_params)
                
                self.buffer.add(obs, action, reward, next_obs, done)
                episode_reward += reward
                obs = next_obs
                state = next_state
                
                self.step_count += 1
                
                # Update network
                if len(self.buffer) >= self.batch_size:
                    batch = self.buffer.sample(self.batch_size)
                    self.params, self.opt_state, loss = self.update_step(
                        self.params, self.target_params, self.opt_state, batch, self.q_network)
                    
                    # Update target network
                    if self.step_count % self.target_update_freq == 0:
                        self.target_params = self.params
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            total_rewards.append(episode_reward)
            if episode % 100 == 0:
                print(f"Episode {episode}, Avg Reward: {np.mean(total_rewards[-100:]):.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return total_rewards

# Usage example
if __name__ == "__main__":
    env = twosteptask.TwoStepTask()
    agent = DQNAgent(env)
    rewards = agent.train(num_episodes=10)