"""Reward sanity check: success rate and reward terms for the scripted expert (driven through the same 3-D
delta action space the agent uses), a random policy and an idle one. A good reward ranks them clearly.

    python tools/check_reward.py --episodes 30
    python tools/check_reward.py --episodes 30 --set environment.posScale=0.03
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.catching import CatchingEnv   # noqa: E402
from utils import loadConfig, applyOverrides  # noqa: E402


class ExpertInDeltaActions:
    """The env's privileged CatchExpert outputs absolute net targets; convert them to clipped [dx, dy, dz] deltas."""
    def __init__(self, catchingEnv):
        task = catchingEnv.env
        task.action_mode = "ee_abs"     # the expert asserts this mode; it only reads state, so switch back right after
        self.expert = task.make_expert()
        task.action_mode = "ee_delta"
        self.task = task

    def __call__(self, observation):
        target = self.expert()[:3]
        return np.clip((target - self.task._target_pos)/self.task.pos_scale, -1, 1)


def run(env, makePolicy, episodes):
    statistics = defaultdict(list)
    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        policy = makePolicy(env)
        episodeReturn, length = 0.0, 0
        while True:
            observation, info = env.step(policy(observation))
            episodeReturn += observation["reward"]
            length += 1
            if observation["isLast"]:
                break
        for key, value in {"return": episodeReturn, "length": length, **info}.items():
            statistics[key].append(value)
    return {key: np.mean(values) for key, values in statistics.items()}


def main(args):
    config = applyOverrides(loadConfig(os.path.join(os.path.dirname(__file__), "..", "configs", "catching.yml")), args.set)
    env = CatchingEnv(config.environment)
    rng = np.random.default_rng(0)
    policies = {"expert": ExpertInDeltaActions,
                "random": lambda env: (lambda observation: rng.uniform(-1, 1, 3)),
                "idle":   lambda env: (lambda observation: np.zeros(3))}
    rows = {name: run(env, makePolicy, args.episodes) for name, makePolicy in policies.items()}
    keys = list(rows["expert"])
    print(f"{'policy':<8}" + "".join(f"{key.replace('reward/', ''):>12}" for key in keys))
    for name, row in rows.items():
        print(f"{name:<8}" + "".join(f"{row[key]:>12.3f}" for key in keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    main(parser.parse_args())
