"""How visible is the ball to the policy camera?

Rolls out the scripted expert in BallisticCatch and, for every candidate camera and image size,
counts the ball's pixels (segmentation render) on every frame of the flight. Also saves a contact
sheet of RGB frames so the views can be compared by eye.

    python tools/camera_check.py --episodes 5 --out cameraCheck
"""
import argparse
import os
import sys

import imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.catching import CAMERAS, makeCatchTask   # noqa: E402
from dynamic_envs.geometry import spherical_camera  # noqa: E402


def setCamera(env, spec):
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    env.model.cam_pos[cid], env.model.cam_quat[cid] = spherical_camera(spec.lookat, spec.distance, spec.azimuth, spec.elevation)
    env.model.cam_fovy[cid] = spec.fovy
    mujoco.mj_camlight(env.model, env.data)  # camera world poses are only recomputed in mj_forward otherwise


def main(args):
    env = makeCatchTask(renderSize=64, actionMode="ee_abs", randomizeCamera=False, randomizeLighting=False)
    ballGeom = env.gid("ball0_geom")
    segRenderers = {size: mujoco.Renderer(env.model, size, size) for size in args.sizes}
    for renderer in segRenderers.values():
        renderer.enable_segmentation_rendering()

    pixels = {(name, size): [] for name in CAMERAS for size in args.sizes}
    sheet = {}
    for episode in range(args.episodes):
        env.reset(seed=episode)
        expert = env.make_expert()
        launchTime = env.launches[0].t
        done, step = False, 0
        while not done:
            _, _, terminated, truncated, info = env.step(expert())
            done = terminated or truncated
            step += 1
            if env.data.time < launchTime:
                continue
            for name, spec in CAMERAS.items():
                setCamera(env, spec)
                for size, renderer in segRenderers.items():
                    renderer.update_scene(env.data, camera="agentview")
                    seg = renderer.render()
                    pixels[(name, size)].append(int(np.sum((seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == ballGeom))))
                if episode == 0 and step % 6 == 0:
                    sheet.setdefault(name, []).append(np.concatenate([np.kron(env.render_camera("agentview", s, s), np.ones((384 // s, 384 // s, 1), np.uint8)) for s in args.sizes], 1))
        print(f"episode {episode}: success={info['success']}")

    print(f"\n{'camera':<12}{'size':>6}{'visible%':>10}{'median px':>11}{'p10 px':>8}")
    for (name, size), counts in pixels.items():
        counts = np.array(counts)
        visible = counts[counts > 0]
        print(f"{name:<12}{size:>6}{100*len(visible)/len(counts):>9.0f}%{np.median(visible) if len(visible) else 0:>11.1f}{np.percentile(visible, 10) if len(visible) else 0:>8.1f}")

    os.makedirs(args.out, exist_ok=True)
    for name, frames in sheet.items():
        imageio.imwrite(os.path.join(args.out, f"{name}.png"), np.concatenate(frames[:6], 0))
    print(f"\nContact sheets (columns = {args.sizes} px, upscaled) written to {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 96])
    parser.add_argument("--out", type=str, default="cameraCheck")
    main(parser.parse_args())
