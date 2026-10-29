"""Catching reward. Every term reads privileged simulator state; the policy never sees it.

    r = catch     * [ball caught this step]
      + progress  * (d_previous - d_now)              from launch on
      + precision * exp(-precisionScale * d_now^2)    from launch on
      - action    * ||a||^2

d is the distance from the net centre to the ball's future path, restricted to the part where the
ball is falling (the net opens upwards) and still above the table. With gravity only (no drag in the
sim) that path is exact until the ball touches something.
"""
import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


def distanceToPath(ballPosition, ballVelocity, point, minHeight, horizon=1.5, dt=1/240):
    t = np.arange(0.0, horizon, dt)[:, None]
    path = ballPosition + ballVelocity*t + 0.5*GRAVITY*t**2
    falling = (ballVelocity[2] + GRAVITY[2]*t[:, 0] < 0) & (path[:, 2] >= minHeight)
    if not falling.any():
        return float(np.linalg.norm(ballPosition - point))
    return float(np.min(np.linalg.norm(path[falling] - point, axis=-1)))


class CatchReward:
    def __init__(self, config, minHeight):
        self.config     = config
        self.minHeight  = minHeight
        self.reset()

    def reset(self):
        self.previousDistance   = None
        self.minDistance        = np.inf
        self.episodeTerms       = {"catch": 0.0, "progress": 0.0, "precision": 0.0, "action": 0.0}

    def __call__(self, env, action, caught):
        terms = {"catch": self.config.catch*float(caught), "progress": 0.0, "precision": 0.0,
                 "action": -self.config.action*float(np.sum(np.square(action)))}

        if env.launched():
            netPosition, _  = env.arm.site_pose()
            distance        = distanceToPath(env.body_pos("ball0"), env.body_linvel("ball0"), netPosition, self.minHeight)
            if self.previousDistance is not None:
                terms["progress"] = self.config.progress*(self.previousDistance - distance)
            terms["precision"]      = self.config.precision*np.exp(-self.config.precisionScale*distance**2)
            self.previousDistance   = distance
            self.minDistance        = min(self.minDistance, distance)

        for key, value in terms.items():
            self.episodeTerms[key] += value
        return sum(terms.values())
