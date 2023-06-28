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

Game = namedtuple('Game', ['reward', 'frames'])
n_games = 10
games = []

for g in range(n_games):
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
  games.append(Game(total_reward, frames))



env.close()

# Save video of the best game
import imageio
best_game = max(games, key=lambda g: g.reward)
print(f"Best game: {best_game.reward}")
imageio.mimsave('./best_game.gif', best_game.frames)
