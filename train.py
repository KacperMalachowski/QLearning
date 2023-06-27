import os
import time
import gymnasium as gym
import tensorflow as tf
from env import ImageEnv
from dqn import DQN

env = gym.make('CarRacing-v2', render_mode='human')
env = ImageEnv(env)

state_dim = (84, 84)
episodes = 1000

agent = DQN(state_dim)
if os.path.isfile('./dqn.h5'):
  agent.load('./dqn.h5')

rewards = 0
time_sum = 0

total_rewards = []

try:
  for i in range(1, episodes + 1):
    print(f"Episode: {i}/{episodes}") 
    agent.tensorboard.step = i
    start_time = time.time()
    state, info = env.reset(seed=4)
    done = False
    index = 0
    total_reward = 0

    while not done:
      action = agent.act(state)
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated


      agent.remember(state, action, reward, next_state, terminated, truncated)

      state = next_state
      index += 1
      total_reward += reward
    
    total_rewards.append(total_reward)

    agent.tensorboard.update_stats(reward=total_reward,reward_min=min(total_rewards), reward_max=max(total_rewards), reward_avg=sum(total_rewards)/len(total_rewards), epsilon=agent.epsilon)
    agent.replay()
finally:
  agent.save("dqn.h5")