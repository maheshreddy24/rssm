"""BallisticCatch (dynamic-environments) with the interface Dreamer expects.

Observations: {"image": (H, W, 3) uint8 from the policy camera, "state": (22,) float32}
              state = gripper position (3), orientation (4), width (1), joint angles (7), joint velocities (7)
Actions:      [dx, dy, dz] in [-1, 1], end-effector position deltas. The net stays facing up and the
              gripper is clamped by the env, so rotation and grip are fixed.
"""
import os
import sys

import numpy as np

try:
    import dynamic_envs
except ImportError:  # not pip-installed: use the copy next to this repo
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dynamic-environments"))
    import dynamic_envs  # noqa: F401
from dynamic_envs.base_env import TABLE_Z, CameraSpec
from dynamic_envs.geometry import spherical_camera
from dynamic_envs.tasks.catching import BallisticCatch
import mujoco

from .rewards import CatchReward

# Policy cameras; compare them with tools/camera_check.py. "paper" is BallisticCatch's own agentview.
# "wide" fits the whole flight (arcs reach 2.3 m) with a larger ball: ~91% of flight frames show it,
# ~4 px at 96x96 (vs 2 px for "paper"). "close" doubles the ball size but loses it near the apex.
CAMERAS = {
    "paper":    CameraSpec(lookat=(0.95, 0.0, TABLE_Z + 0.1),  distance=2.9, azimuth=215, elevation=-22),
    "wide":     CameraSpec(lookat=(1.00, 0.0, TABLE_Z + 0.6),  distance=2.6, azimuth=235, elevation=-10),
    "close":    CameraSpec(lookat=(0.80, 0.0, TABLE_Z + 0.45), distance=2.0, azimuth=215, elevation=-18),
}
VIDEO_CAMERA = CameraSpec(lookat=(0.95, 0.0, TABLE_Z + 0.3), distance=3.0, azimuth=225, elevation=-15)


def makeCatchTask(camera="wide", posScale=0.04, renderSize=96, actionMode="ee_delta", randomizeCamera=False, randomizeLighting=True, **taskKwargs):
    taskClass = type("BallisticCatch", (BallisticCatch,), {"camera": CAMERAS[camera], "pos_scale": posScale})
    return taskClass(render_size=renderSize, action_mode=actionMode, obs_mode="rgb_state",
                     randomize_camera=randomizeCamera, randomize_lighting=randomizeLighting, **taskKwargs)


class CatchingEnv:
    def __init__(self, config, seed=None):
        self.env = makeCatchTask(config.camera, config.posScale, config.imageSize, randomizeCamera=config.randomizeCamera,
                                 randomizeLighting=config.randomizeLighting, **(config.get("task") or {}))
        self.reward = CatchReward(config.reward, minHeight=TABLE_Z)
        self.seed   = seed
        self.done   = True

        self.observationShapes  = {"image": (config.imageSize, config.imageSize, 3), "state": (22,)}
        self.actionSize         = 3

    def reset(self, seed=None):
        observation, _ = self.env.reset(seed=self.seed if seed is None else seed)
        self.seed = None  # seed only the first episode; later ones continue the env's own RNG stream
        videoCamera = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        self.env.model.cam_pos[videoCamera], self.env.model.cam_quat[videoCamera] = spherical_camera(
            VIDEO_CAMERA.lookat, VIDEO_CAMERA.distance, VIDEO_CAMERA.azimuth, VIDEO_CAMERA.elevation)
        mujoco.mj_camlight(self.env.model, self.env.data)
        self.reward.reset()
        self.done = False
        return self.makeObservation(observation, 0.0, isFirst=True), {}

    def step(self, action):
        if self.done:
            return self.reset()
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        fullAction = np.concatenate([action, np.zeros(3), [-1.0]])  # no rotation, gripper ignored
        observation, _, terminated, truncated, info = self.env.step(fullAction)
        reward = self.reward(self.env, action, info["success"])
        self.done = terminated or truncated

        episodeInfo = {}
        if self.done:
            episodeInfo = {"success": float(info["success"]), "minDistance": self.reward.minDistance,
                           **{f"reward/{key}": value for key, value in self.reward.episodeTerms.items()}}
        return self.makeObservation(observation, reward, isLast=self.done, isTerminal=terminated), episodeInfo

    def makeObservation(self, observation, reward, isFirst=False, isLast=False, isTerminal=False):
        return {"image"      : observation["image"],
                "state"      : np.concatenate([observation["state"], self.env.arm.qd]).astype(np.float32),
                "reward"     : np.float32(reward),
                "isFirst"    : isFirst,
                "isLast"     : isLast,
                "isTerminal" : isTerminal}

    def render(self, size=256):
        return self.env.render_camera("overview", size, size)

    def close(self):
        self.env.close()
