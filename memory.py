import numpy as np

class ReplayMemory(object):
    def __init__(
      self, 
      input_shape,
      action_number,
      maxlen=500,
      discrete=False,
    ):
      self.mem_size = maxlen
      self.mem_counter = 0
      self.discrete = discrete
      self.state_mem = np.zeros((self.mem_size, input_shape))
      self.new_state_mem = np.zeros((self.mem_size, input_shape))
      dtype = np.int8 if self.discrete else np.float32
      self.action_mem = np.zeros((self.mem_size, action_number), dtype=dtype)
      self.reward_mem = np.zeros(self.mem_size)
      self.terminal_mem = np.zeros(self.mem_size, dtype=np.float32)

    def append(self, state, action, reward, new_state, done):
      index = self.mem_counter % self.mem_size
      self.state_mem[index] = state
      self.new_state_mem[index] = new_state
      self.reward_mem[index] = reward
      self.terminal_mem[index] = 1 - int(done)
      if self.discrete:
        actions = np.zeros(self.action_mem.shape[1])
        actions[action] = 1.0
        self.action_mem[index] = actions
      else:
        self.action_mem[index] = action
      self.mem_counter += 1
       
    def sample(self, batch_size):
      max_mem = min(self.mem_counter, self.mem_size)
      batch = np.random.choice(max_mem, batch_size)

      states = self.state_mem[batch]
      new_states = self.new_state_mem[batch]
      rewards = self.reward_mem[batch]
      actions = self.action_mem[batch]
      terminal = self.terminal_mem[batch]

      return states, actions, rewards, new_states, terminal