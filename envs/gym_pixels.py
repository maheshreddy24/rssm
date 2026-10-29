import gymnasium as gym
import numpy as np
from PIL import Image


class GymPixelsEnv:
    """A gymnasium env with image observations and a Box action space (e.g. CarRacing-v3), resized to
    imageSize x imageSize. Actions in [-1, 1] are mapped to the env's own bounds."""
    def __init__(self, config, seed=None):
        self.env        = gym.make(config.name, render_mode="rgb_array")
        self.imageSize  = config.imageSize
        self.seed       = seed
        self.done       = True
        assert isinstance(self.env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
        self.actionLow, self.actionHigh = self.env.action_space.low, self.env.action_space.high
        self.observationShapes  = {"image": (self.imageSize, self.imageSize, 3)}
        self.actionSize         = self.env.action_space.shape[0]

    def reset(self, seed=None):
        observation, _ = self.env.reset(seed=self.seed if seed is None else seed)
        self.seed = None
        self.done = False
        return self.makeObservation(observation, 0.0, isFirst=True), {}

    def step(self, action):
        if self.done:
            return self.reset()
        action = self.actionLow + (np.clip(action, -1, 1) + 1)/2*(self.actionHigh - self.actionLow)
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.done = terminated or truncated
        return self.makeObservation(observation, reward, isLast=self.done, isTerminal=terminated), {}

    def makeObservation(self, observation, reward, isFirst=False, isLast=False, isTerminal=False):
        image = np.asarray(Image.fromarray(observation).resize((self.imageSize, self.imageSize), Image.BILINEAR))
        return {"image": image, "reward": np.float32(reward), "isFirst": isFirst, "isLast": isLast, "isTerminal": isTerminal}

    def render(self, size=None):
        return self.env.render()

    def close(self):
        self.env.close()
