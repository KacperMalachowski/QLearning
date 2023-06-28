import gc
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from memory import ReplayMemory


class DQN:
  def __init__(
    self,
    state_dim,
    action_number,
    learning_rate=0.0001,
    epsilon = 1.0,
    epsilon_min=0.01,
    epsilon_decay=0.999,
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
    
  
  def build_model(self, state_dim, action_number):
    model = Sequential([
       Dense(256, input_shape=(state_dim, )),
       Activation('relu'),
       Dense(256),
       Activation('relu'),
       Dense(action_number)
    ])
    model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
    return model
  
  def act(self, state, training=True):
    state = state[np.newaxis, :]
    rand = np.random.random()

    if rand < self.epsilon:
      action = np.random.choice(self.action_space)
    else:
      actions = self.network.predict(state, verbose=0)
      action = np.argmax(actions)

    return action
  
  def remember(self, state, action, reward, next_state, terminated, truncated):
    self.memory.append(state, action, reward, next_state, terminated or truncated)

  def replay(self):
    if self.memory.mem_counter < self.batch_size:
      return
    
    state, action, reward, new_state, done = self.memory.sample(self.batch_size)

    action_values = np.array(self.action_space, dtype=np.int8)
    action_indices = np.dot(action, action_values)

    q_eval = self.network.predict(state, verbose=0)
    q_next = self.network.predict(new_state, verbose=0)

    q_target =  q_eval.copy()

    batch_index = np.arange(self.batch_size, dtype=np.int32)

    q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done

    self.network.fit(state, q_target, verbose=0)
    keras.backend.clear_session()

    self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
  
  def load(self, name, ignore_epsilon=False):
    with open(name, "rb") as f:
      store = pickle.load(f)

      self.network.set_weights(store['Train'])
      
      if not ignore_epsilon:
        self.epsilon = store['Epsilon']

  def save(self, name):
    store = {
      'Train': self.network.get_weights(),
      'Epsilon': self.epsilon
    }

    with open(name, "wb") as f:
      pickle.dump(store, f)