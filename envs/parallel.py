import contextlib
import multiprocessing
import sys
import numpy as np


def _worker(pipe, makeEnv):
    env = makeEnv()
    pipe.send((env.observationShapes, env.actionSize))
    while True:
        command, data = pipe.recv()
        if command == "step":
            pipe.send(env.step(data))
        elif command == "reset":
            pipe.send(env.reset())
        elif command == "close":
            env.close()
            pipe.close()
            break


@contextlib.contextmanager
def _withoutMainScript():
    """spawn re-imports the parent's script (main.py, and with it torch, pandas, plotly) in every worker, ~500 MB each.
    Workers only need the env, which is imported by module name, so hide the script while they start."""
    main = sys.modules["__main__"]
    saved = {key: main.__dict__.pop(key) for key in ("__file__", "__spec__") if key in main.__dict__}
    try:
        yield
    finally:
        main.__dict__.update(saved)


class ParallelEnvs:
    """Steps len(makeEnvs) environments in lockstep, each in its own process (or in this process when parallel=False),
    and stacks their observations along a leading env axis. An env that finished resets on its next step."""
    def __init__(self, makeEnvs, parallel=True):
        self.numEnvs    = len(makeEnvs)
        self.parallel   = parallel
        if parallel:
            context     = multiprocessing.get_context("spawn")  # each process gets its own clean MuJoCo / EGL context
            self.pipes  = []
            with _withoutMainScript():
                for makeEnv in makeEnvs:
                    parentPipe, childPipe = context.Pipe()
                    context.Process(target=_worker, args=(childPipe, makeEnv), daemon=True).start()
                    self.pipes.append(parentPipe)
            self.observationShapes, self.actionSize = [pipe.recv() for pipe in self.pipes][0]
        else:
            self.envs = [makeEnv() for makeEnv in makeEnvs]
            self.observationShapes, self.actionSize = self.envs[0].observationShapes, self.envs[0].actionSize

    def _call(self, command, data=None):
        if self.parallel:
            for i, pipe in enumerate(self.pipes):
                pipe.send((command, None if data is None else data[i]))
            results = [pipe.recv() for pipe in self.pipes]
        else:
            results = [env.step(data[i]) if command == "step" else env.reset() for i, env in enumerate(self.envs)]
        observations, infos = zip(*results)
        return {key: np.stack([observation[key] for observation in observations]) for key in observations[0]}, list(infos)

    def reset(self):
        return self._call("reset")

    def step(self, actions):
        return self._call("step", actions)

    def close(self):
        if self.parallel:
            for pipe in self.pipes:
                pipe.send(("close", None))
        else:
            for env in self.envs:
                env.close()
