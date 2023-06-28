import gc
import os
import time
import gymnasium as gym
import numpy as np
from dqn import DQN
from utils import plotLearning

env = gym.make('LunarLander-v2')

agent = DQN(
  state_dim=8, 
  action_space=env.action_space, 
  gamma=0.99, 
  epsilon=1.0,
  learning_rate=0.0001,
  buffer_size=2000,
  batch_size=128,
  epsilon_min=0.05,
  epsilon_decay=0.99
)
if os.path.isfile('./dqn.h5'):
  agent.load('./dqn.h5')

rewards = 0
time_sum = 0

total_rewards = []
last_100_rewards = np.zeros((100,), dtype=float)
eps_history = []
i = 1
episodes = 500
avg_reward = 0
try:
  while np.mean(last_100_rewards) < 200:
    print(f"Episode: {i}") 
    start_time = time.time()
    state, info = env.reset(seed=4)
    state = agent.preprocess_state(state)
    done = False
    total_reward = 0

    if i % 100 == 0:
      last_100_rewards = np.zeros((100,), dtype=float)

    while not done:
      action = agent.act(state)
      next_state, reward, terminated, truncated, _ = env.step(action.item())
      done = terminated or truncated

      next_state = agent.preprocess_state(next_state)

      agent.remember(state, action, reward, next_state)

      if done:
        next_state = None

      state = next_state
      total_reward += reward

      agent.replay()
    
    total_rewards.append(total_reward)
    last_100_rewards[i % 100] = total_reward
    eps_history.append(agent.epsilon)

    avg_reward = np.mean(total_rewards[max(0, i-100): (i+1)])
    print(" Reward: %.2f" % total_reward, "\n Avg Reward: %.2f" % avg_reward, "\n Last 100 Avg Reward: %.2f" % np.mean(last_100_rewards))

    i += 1

  x = [j + 1 for j in range(episodes)]
  plotLearning(x, total_rewards, eps_history, "lunar.png")

  print(f"Solved in: {i} episodes!")

finally:
  agent.save("dqn.h5")