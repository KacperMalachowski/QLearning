from collections import deque, namedtuple
import gc
import math
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNModel(nn.Module):
  def __init__(self, state_dim, action_number):
    super(DQNModel, self).__init__()
    self.fc1 = nn.Linear(state_dim, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, action_number)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN:
  def __init__(
    self,
    state_dim,
    action_space,
    learning_rate=0.0001,
    epsilon = 1.0,
    epsilon_min=0.01,
    epsilon_decay=0.99,
    gamma=0.99,
    batch_size=128,
    buffer_size=2000,
  ): 
    self.action_space = action_space
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    
    self.memory = deque([], maxlen=buffer_size)

    self.network = self.build_model(state_dim, 4)
    self.target_net = self.build_model(state_dim, 4)
    self.update_target_net()
    self.target_net.eval()

    self.optimizer = optim.AdamW(self.network.parameters(), lr=self.learning_rate, amsgrad = True)

    self.steps = 0
    
    self.loss_fn = nn.SmoothL1Loss()
  
  def build_model(self, state_dim, action_number):
    model = DQNModel(state_dim, action_number)
    model.to(device)
    return model
  
  def act(self, state):
    sample = random.random()

    epsilon_trsh = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1. * self.steps / self.epsilon_decay)

    self.steps += 1

    if sample < epsilon_trsh:
      action = torch.tensor([[self.action_space.sample()]], device = device, dtype=torch.long)
    else:
      with torch.no_grad():
        action = self.network(state).max(1)[1].view(1, 1)

    return action
  
  def preprocess_state(self, state):
    if isinstance(state, np.ndarray):
      state = tuple(state)
    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
    return state
  
  def remember(self, state, action, reward, next_state):
    reward = torch.tensor([reward], device = device)
    self.memory.append(Transition(state, action, next_state, reward))

  def replay(self):
    if len(self.memory) < self.batch_size:
      return
    
    transitions = random.sample(self.memory,self.batch_size) 
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = self.network(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(self.batch_size, device=device)
    with torch.no_grad():
      next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

    expected_state_action = (next_state_values * self.gamma) + reward_batch

    loss = self.loss_fn(state_action_values, expected_state_action.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
    self.optimizer.step()

    self.update_target_net(0.001)

    self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
    
  def update_target_net(self, tau = 1.0):
    for target_param, local_param in zip (self.target_net.parameters(), self.network.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

  def load(self, name, ignore_epsilon=False):
    checkpoint = torch.load(name)
    self.network.load_state_dict(checkpoint['model_state_dict'])

    if not ignore_epsilon:
      self.epsilon = checkpoint['epsilon']

  def save(self, name):
    checkpoint = {
      'model_state_dict': self.network.state_dict(),
      'epsilon': self.epsilon
    }

    torch.save(checkpoint, name)