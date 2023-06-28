# Run agent with trained weights and save video
import gymnasium as gym
import numpy as np
from dqn import DQN

env = gym.make('LunarLander-v2', render_mode='rgb_array')

agent = DQN(
  state_dim=8,
  action_space=env.action_space
)

agent.load('./dqn.h5')

state, info = env.reset(seed=4)
state = agent.preprocess_state(state)

done = False
total_reward = 0
frames = []

while not done:
  action = agent.act(state)
  next_state, reward, terminated, truncated, _ = env.step(action.item())
  done = terminated or truncated

  next_state = agent.preprocess_state(next_state)

  if done:
    next_state = None

  state = next_state
  total_reward += reward
  frames.append(env.render())

print(f"Total reward: {total_reward}")

env.close()

# Save video
import imageio
imageio.mimsave('./lunar_lander.gif', frames)