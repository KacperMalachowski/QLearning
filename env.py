import gymnasium as gym
import numpy as np
import cv2

class ImageEnv(gym.Wrapper):
  def __init__(
    self,
    env,
    skip_frames=4,
    stack_frames=4,
    initial_no_op=50,
    **kwargs
  ):
    super(ImageEnv, self).__init__(env, **kwargs)
    self.initial_no_op = initial_no_op
    self.skip_frames = skip_frames
    self.stack_frames = stack_frames

  def _preprocess(self, img):
    img = cv2.resize(img, dsize=(84,84))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

  def reset(self, **kwargs):
    s, info = self.env.reset(**kwargs)

    for i in range(self.initial_no_op):
      s, r, term, trunc, info = self.env.step((0, 0, 0))

    s = self._preprocess(s)

    self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))
    return self.stacked_state, info
  
  def step(self, action):
    reward = 0

    for _ in range(self.skip_frames):
      s, r, term, trunc, info = self.env.step(action)
      done = term or trunc
      reward += r

      if done:
        break

    s = self._preprocess(s)

    self.stacked_state = np.concatenate((self.stacked_state, s[np.newaxis]), axis=0)

    return self.stacked_state, reward, term, trunc, info