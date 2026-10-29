"""Environments with the interface Dreamer uses:

    observation, info = env.reset()
    observation, info = env.step(action)     # action in [-1, 1]^actionSize; after isLast it resets instead
    observation = {<inputs in env.observationShapes>, "reward", "isFirst", "isLast", "isTerminal"}
    info        = {} during an episode, episode statistics (e.g. "success") on its last step
"""
from functools import partial


def makeEnvironment(environmentConfig, seed=None):
    if environmentConfig.name == "BallisticCatch":
        from .catching import CatchingEnv
        return CatchingEnv(environmentConfig, seed)
    from .gym_pixels import GymPixelsEnv
    return GymPixelsEnv(environmentConfig, seed)


def environmentFactories(environmentConfig, numEnvs, seed):
    """Picklable constructors for the parallel workers, each env with its own seed."""
    return [partial(makeEnvironment, environmentConfig, seed + i) for i in range(numEnvs)]
