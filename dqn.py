import gc
import pickle
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

class DQN:
  def __init__(
    self,
    state_dim,
    action_number,
    learning_rate=0.0001,
    epsilon = 1.0,
    epsilon_min=0.01,
    epsilon_decay=0.99,
    gamma=0.99,
    batch_size=128,
    buffer_size=2000,
  ): 
    self.action_space = [i for i in range(action_number)]
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    
    self.memory = ReplayMemory(state_dim, action_number, discrete=True, maxlen=buffer_size)

    self.network = self.build_model(state_dim, action_number)
    self.target_net = self.build_model(state_dim, action_number)
    self.update_target_net()
    self.target_net.eval()

    self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
    
    self.loss_fn = nn.MSELoss()
  
  def build_model(self, state_dim, action_number):
    model = DQNModel(state_dim, action_number)
    model.to(device)
    return model
  
  def act(self, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    rand = torch.rand(1).item()

    if rand < self.epsilon:
      action = torch.randint(0, len(self.action_space), (1,)).item()
    else:
      with torch.no_grad():
        action = self.network(state).argmax(1).item()

    return action
  
  def remember(self, state, action, reward, next_state, terminated, truncated):
    self.memory.append(state, action, reward, next_state, terminated or truncated)

  def replay(self):
    if self.memory.mem_counter < self.batch_size:
      return
    
    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    q_eval = self.network(states)
    q_next = self.target_net(next_states)
    q_target = rewards + self.gamma * q_next * (1 - dones)

    loss = self.loss_fn(q_eval, q_target)

    self.optimizer.zero_grad()
    loss.backward()
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