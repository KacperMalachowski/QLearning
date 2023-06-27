import os
import pickle
import random
import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque

log_dir="logs/{}-{}".format("DQN", int(time.time()))

class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
       pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=False):
       pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class DQN:
  def __init__(
    self,
    state_dim,
    action_dim = [
      (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),
      (-1, 1, 0), (0, 1, 0), (1, 1, 0),
      (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),
      (-1, 0, 0), (0, 0, 0), (1, 0, 0)
    ],
    learning_rate=0.001,
    epsilon = 1.0,
    epsilon_min=0.01,
    epsilon_decay=0.99,
    gamma=0.95,
    batch_size=32,
    warmup_steps=1000,
    buffer_size=2500,
    target_update_interval=1000,
  ): 
    self.action_dim = action_dim
    self.epsilon = epsilon
    self.gamma = gamma
    self.batch_size = batch_size
    self.warmup_steps = warmup_steps
    self.target_update_interval = target_update_interval

    self.learning_rate = learning_rate
    self.network = self.build_model(state_dim, action_dim)
    self.target_network = self.build_model(state_dim, action_dim)
    self._update_target_model(tau=1.0)

    self.buffer = deque(maxlen=buffer_size)

    self.total_steps = 0
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay

    self.tensorboard = ModifiedTensorBoard("DQN", log_dir=log_dir)
    
  def build_model(self, state_dim, action_dim):
    model = Sequential()
    model.add(Reshape(target_shape=(*state_dim, 1), input_shape=state_dim, name='layers_reshape'))
    model.add(Flatten(name='layes_flatten'))
    model.add(Dense(5, activation='relu', name = 'layers_dense'))
    model.add(Dense(len(action_dim), activation='relu', name = 'layers_action_dense'))
    # model.add(Dense(24, activation='relu'))
    # model.add(Dense(24, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(len(action_dim), activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7), metrics=['accuracy'])
    return model
  
  def act(self, state, training=True):
    if training and np.random.rand() < self.epsilon:
      action = random.randrange(len(self.action_dim))
    else:
      action_probs = self.network.predict(state, verbose=0)
      action = np.argmax(action_probs[0])
    return self.action_dim[action]
  
  def remember(self, state, action, reward, next_state, terminated, truncated):
    self.buffer.append((state, self.action_dim.index(action), reward, next_state, terminated, truncated))
    self.total_steps += 1

  def replay(self):
    if len(self.buffer) < self.batch_size:
      return
    
    sample_batch = random.sample(self.buffer, self.batch_size)

    states = []
    targets = []

    for state, action, reward, next_state, terminated, truncated in sample_batch:
      target = self.network.predict(state, verbose=0)[0]
      if terminated or truncated:
        target[action] = reward
      else:
        next_q_values = self.target_network.predict(next_state, verbose=0)[0]
        target[action] = reward + self.gamma * np.amax(next_q_values)

      states.append(state[0])
      targets.append(target)

    self.network.fit(
      tf.stack(states),
      tf.stack(targets),
      epochs=1,
      verbose=0,
      callbacks=[self.tensorboard]
    )

    if self.total_steps % self.target_update_interval == 0:
      self._update_target_model(tau=0.01)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
  
  def _update_target_model(self, tau):
    for target_weight, online_weight in zip(self.target_network.trainable_variables, self.network.trainable_variables):
       target_weight.assign(target_weight * (1.0 - tau) + online_weight * tau)
  
  def load(self, name, ignore_epsilon=False):
    with open(name, "rb") as f:
      store = pickle.load(f)

      self.network.set_weights(store['Train'])
      self.target_network.set_weights(store['Weights'])
      
      if not ignore_epsilon:
        self.steps = store['Steps']
        self.epsilon = store['Epsilon']

  def save(self, name):
    store = {
      'Train': self.network.get_weights(),
      'Weights': self.target_network.get_weights(),
      'Steps': self.total_steps,
      'Epsilon': self.epsilon
    }

    with open(name, "wb") as f:
      pickle.dump(store, f)