import gc
import os
import time
import gymnasium as gym
import numpy as np
from dqn import DQN
from utils import plotLearning

env = gym.make('LunarLander-v2', render_mode='human')

agent = DQN(
  state_dim=8, 
  action_number=4, 
  gamma=0.99, 
  epsilon=1.0,
  learning_rate=0.0005,
  buffer_size=1000,
  batch_size=64,
  epsilon_min=0.01
)
if os.path.isfile('./dqn.h5'):
  agent.load('./dqn.h5')

rewards = 0
time_sum = 0

total_rewards = []
eps_history = []
i = 1
episodes = 500
try:
  while i < episodes:
    print(f"Episode: {i}/{episodes}") 
    start_time = time.time()
    state, info = env.reset(seed=4)
    done = False
    total_reward = 0

    while not done:
      action = agent.act(state)
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated


      agent.remember(state, action, reward, next_state, terminated, truncated)

      state = next_state
      total_reward += reward

      agent.replay()
    
    total_rewards.append(total_reward)
    eps_history.append(agent.epsilon)

    avg_reward = np.mean(total_rewards[max(0, i-100): (i+1)])
    print(" Reward: %.2f" % total_reward, "Avg Reward: %.2f" % avg_reward)

    if i % 10 == 0 and i > 0:
      x = [j for j in range(i)]
      plotLearning(x, total_rewards, eps_history, f"lunar_{i}.png")

    i += 1

  x = [x + 1 for i in range(episodes)]
  plotLearning(x, total_rewards, eps_history, "lunar.png")


finally:
  agent.save("dqn.h5")