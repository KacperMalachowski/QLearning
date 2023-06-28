import random

class ReplayMemory():
    def __init__(
      self, 
      maxlen=500
    ):
      self.maxlen = maxlen
      self.memory = []

    def append(self, data):
      if len(self.memory) > self.maxlen:
        index = random.randint(0, self.maxlen)
        self.memory[index] = data
      else:
        self.memory.append(data)

    def sample(self, batch_size):
      batch = []
      for _ in range(batch_size):
        batch.append(self.memory.pop())

      return batch
      

    def pop(self):
      return self.memory.pop()
    
    def __len__(self):
      return len(self.memory)
        