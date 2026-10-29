import numpy as np
import torch


class ReplayBuffer:
    """One ring buffer per environment. Step t stores the observation x_t, the action a_t taken at it, the reward r_t
    received on arriving at it, and the isFirst / isTerminal flags. Images stay uint8 until they reach the encoder.

    Sampled sequences are contiguous in time within one environment and may span several episodes; the world
    model resets its state wherever isFirst is set, so no padding or episode bookkeeping is needed."""
    def __init__(self, observationShapes, actionSize, numEnvs, capacity, device):
        self.numEnvs        = numEnvs
        self.capacity       = int(capacity)//numEnvs
        self.device         = device
        self.observationKeys = list(observationShapes)

        dtypes = {"image": np.uint8}
        self.data = {key: np.zeros((numEnvs, self.capacity, *shape), dtype=dtypes.get(key, np.float32)) for key, shape in observationShapes.items()}
        self.data |= {"action"      : np.zeros((numEnvs, self.capacity, actionSize), dtype=np.float32),
                      "reward"      : np.zeros((numEnvs, self.capacity), dtype=np.float32),
                      "isFirst"     : np.zeros((numEnvs, self.capacity), dtype=bool),
                      "isTerminal"  : np.zeros((numEnvs, self.capacity), dtype=bool)}
        self.index  = 0
        self.size   = 0

    def __len__(self):
        return self.size*self.numEnvs

    def add(self, observations, actions):
        """observations: dict of arrays with a leading env axis (as returned by the envs), actions: (numEnvs, actionSize)."""
        for key in self.observationKeys + ["reward", "isFirst", "isTerminal"]:
            self.data[key][:, self.index] = observations[key]
        self.data["action"][:, self.index] = actions
        self.index  = (self.index + 1) % self.capacity
        self.size   = min(self.size + 1, self.capacity)

    def sample(self, batchSize, sequenceLength):
        assert self.size >= sequenceLength, "not enough data in the buffer to sample"
        envs    = np.random.randint(0, self.numEnvs, batchSize)
        starts  = np.random.randint(0, self.size - sequenceLength + 1, batchSize)
        if self.size == self.capacity:
            starts = starts + self.index  # oldest step sits at the write index once the buffer wraps
        indices = (starts[:, None] + np.arange(sequenceLength)[None]) % self.capacity
        return {key: torch.as_tensor(value[envs[:, None], indices]).to(self.device, non_blocking=True) for key, value in self.data.items()}
