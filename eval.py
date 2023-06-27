import os
import time
import gymnasium as gym
from env import ImageEnv
from dqn import DQN

env = gym.make('CarRacing-v2', render_mode='human')
env = ImageEnv(env)

state_dim = (84, 84)
episodes = 1000

agent = DQN(state_dim, epsilon = 0)
if os.path.isfile('./dqn.h5'):
  agent.load('./dqn.h5', True)

rewards = 0
time_sum = 0
episodes = 20

for i in range(episodes):
  print(f"Episode: {i}/{episodes}")
  start_time = time.time()
  state, info = env.reset()

  total_reward = 0
  done = False
  time_frame_count = 1

  while not done:
    action = agent.act(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    total_reward += reward

    state = next_state
    time_frame_count += 1
  
  end_time = time.time() - start_time
  time_sum += end_time
  print(f" Reward: {total_reward}\n Avg Reward: {rewards / i + 1}\n Steps: {agent.total_steps}\n Avg Steps: {agent.total_steps / i + 1}\n Time: {end_time}s\n Avg Time: {time_sum / i}\n Epsilon: {agent.epsilon}")

  