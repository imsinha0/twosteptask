import gym
import numpy as np
import random
from collections import defaultdict
import twosteptask
import jax

def run_q_learning():
    env = twosteptask.TwoStepTask()
    params = env.default_params
    
    main_key = jax.random.PRNGKey(0)
    
    Q = np.zeros((params.num_states, 2)) 

    alpha = 0.1       # learning rate
    gamma = 0.99      # discount factor
    epsilon = 0.1     # exploration rate
    episodes = 1000

    def choose_action(state_idx, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)  # random action
        else:
            return np.argmax(Q[state_idx])

    total_rewards = []
    
    for ep in range(episodes):
        main_key, reset_key = jax.random.split(main_key)
        
        _, state = env.reset(key=reset_key, params=params)
        done = False
        episode_reward = 0
        
        while not done:
            state_idx = state.state_idx
            action = choose_action(state_idx, epsilon)
            
            main_key, step_key = jax.random.split(main_key)
            
            _, next_state, reward, done, _ = env.step(
                key=step_key,
                state=state,
                action=action,
                params=params
            )
            
            next_state_idx = next_state.state_idx
            best_next_action = np.argmax(Q[next_state_idx])
            td_target = reward + gamma * Q[next_state_idx][best_next_action]
            td_error = td_target - Q[state_idx][action]
            Q[state_idx][action] += alpha * td_error
            
            state = next_state
            episode_reward += reward
        
        epsilon = max(0.01, epsilon * 0.999)
        
        total_rewards.append(episode_reward)
        
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-1000:])
            print(f"Episode {ep+1}, Average Reward (last 1000): {avg_reward:.4f}, Epsilon: {epsilon:.4f}")

    print("\nLearned Q-values:")
    for state_idx in range(params.num_states):
        print(f"State {state_idx}: {Q[state_idx]}")

    print("\nTesting the learned policy:")
    main_key, test_key = jax.random.split(main_key)
    obs, state = env.reset(key=test_key, params=params)
    done = False
    total_reward = 0
    
    
    step_count = 0
    while not done and step_count < 10: 
        state_idx = state.state_idx
        action = np.argmax(Q[state_idx]) 
        print(f"Taking action: {action}")
        
        main_key, step_key = jax.random.split(main_key)
        _, state, reward, done, _ = env.step(key=step_key, state=state, action=action, params=params)
        
        total_reward += reward
        step_count += 1

    print("Total reward:", total_reward)
    return Q

if __name__ == "__main__":
    Q = run_q_learning()